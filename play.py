#!/usr/bin/env python3
"""Minimal pygame-ce play/debug shell for the shared Chain Reaction core.

This module deliberately owns presentation only. Legal moves, state mutation,
cascades, and winners all come from ``chain_reaction_core.PyChainReaction``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import pygame
except ImportError as exc:  # pragma: no cover - exercised by humans.
    raise SystemExit(
        "pygame-ce is not installed. Run with `uv run --group play python play.py` "
        "or install the `play` dependency group."
    ) from exc


ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

try:
    import chain_reaction_core
except ImportError as exc:  # pragma: no cover - exercised by humans.
    raise SystemExit(
        "Could not import the Cython core bridge. Build it first:\n"
        "  cd training\n"
        "  uv run --group dev setup.py build_ext --inplace\n"
        "Then run from the repository root:\n"
        "  uv run --group play python play.py"
    ) from exc


BOARD_SIZE = 8
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
PANEL_WIDTH = 280
GRID_MARGIN = 28
STATUS_HEIGHT = 72
FPS = 60

BACKGROUND = (18, 20, 26)
PANEL = (28, 31, 40)
GRID = (71, 77, 92)
EMPTY = (36, 40, 51)
TEXT = (230, 234, 242)
MUTED = (144, 153, 171)
LEGAL = (86, 172, 111)
ILLEGAL = (95, 55, 65)
PLAYER_COLORS = {
    0: EMPTY,
    1: (77, 154, 245),
    2: (239, 93, 98),
}
PLAYER_NAMES = {
    1: "Blue / P1",
    2: "Red / P2",
}
CHECKPOINT_ROOTS = (
    ROOT / "training" / "checkpoints",
    ROOT / "checkpoints",
    ROOT / "models",
)


@dataclass(frozen=True)
class BoardSnapshot:
    tokens: tuple[int, ...]
    owners: tuple[int, ...]
    exploded: tuple[int, ...] = (0,) * CELL_COUNT
    label: str = ""


def current_player(env: object) -> int:
    """Return the alternating player to move from core-owned turn count."""
    return 1 if env.turn_count % 2 == 0 else 2


def make_snapshot(env: object, label: str = "") -> BoardSnapshot:
    return BoardSnapshot(tuple(env.tokens), tuple(env.owners), label=label)


def wave_snapshots(env: object) -> list[BoardSnapshot]:
    frames: list[BoardSnapshot] = []
    wave_count = int(env.wave_count)
    tokens = env.wave_tokens
    owners = env.wave_owners
    exploded = env.wave_exploded

    for wave in range(wave_count):
        start = wave * CELL_COUNT
        stop = start + CELL_COUNT
        frames.append(
            BoardSnapshot(
                tuple(tokens[start:stop]),
                tuple(owners[start:stop]),
                tuple(exploded[start:stop]),
                label=f"wave {wave + 1}/{wave_count}",
            )
        )
    return frames


def resolve_checkpoint(value: str | None) -> Path | None:
    if value is None:
        return None
    if value != "latest":
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    candidates = sorted(
        (path for root in CHECKPOINT_ROOTS if root.exists() for path in root.rglob("*.pt")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        roots = ", ".join(str(root) for root in CHECKPOINT_ROOTS)
        raise SystemExit(f"No Torch checkpoints found under: {roots}")
    return candidates[0]


class TorchPolicy:
    def __init__(self, checkpoint_path: Path, temperature: float, device: str) -> None:
        try:
            import torch
            from training.torch_ppo.model import ChainReactionNet
        except ImportError as exc:  # pragma: no cover - exercised by humans.
            raise SystemExit(
                "Torch policy runtime is not installed. Run with:\n"
                "  uv run --group play --group ai python play.py --checkpoint latest\n"
                "or use `--group training` if you already keep the training stack installed."
            ) from exc

        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")

        self.torch = torch
        self.temperature = temperature
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        board_size = int(config.get("board_size", checkpoint.get("board_size", BOARD_SIZE)))
        if board_size != BOARD_SIZE:
            raise SystemExit(
                f"checkpoint board_size={board_size} is not compatible with this {BOARD_SIZE}x{BOARD_SIZE} shell"
            )
        self.checkpoint_path = checkpoint_path
        self.step = checkpoint.get("step")
        self.model = ChainReactionNet(board_size=BOARD_SIZE)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(20260510)

    @property
    def label(self) -> str:
        name = self.checkpoint_path.name
        if self.step is None:
            return name
        return f"{name} @ {self.step}"

    def select_action(self, env: object, player: int) -> int:
        torch = self.torch
        observation = torch.tensor([env.observation(player)], dtype=torch.float32, device=self.device)
        legal = torch.tensor(env.legal_actions(player), dtype=torch.bool, device=self.device)
        if not bool(legal.any().item()):
            raise RuntimeError(f"core returned no legal actions for player {player}")

        with torch.no_grad():
            logits, _value = self.model(observation)
            masked = logits[0].masked_fill(~legal, torch.finfo(logits.dtype).min)
            if self.temperature <= 0.0:
                return int(masked.argmax().item())
            probabilities = torch.softmax(masked / self.temperature, dim=0)
            return int(torch.multinomial(probabilities, 1, generator=self.generator).item())


class ChainReactionApp:
    def __init__(
        self,
        cell_size: int,
        wave_ms: int,
        policy: TorchPolicy | None = None,
        ai_player: int = 2,
        ai_delay_ms: int = 300,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Chain Reaction — core-backed playable shell")

        self.cell_size = cell_size
        self.board_px = BOARD_SIZE * cell_size
        self.width = self.board_px + GRID_MARGIN * 3 + PANEL_WIDTH
        self.height = self.board_px + GRID_MARGIN * 2 + STATUS_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.big_font = pygame.font.Font(None, 38)

        self.board_rect = pygame.Rect(GRID_MARGIN, GRID_MARGIN, self.board_px, self.board_px)
        self.env = chain_reaction_core.PyChainReaction()
        self.policy = policy
        self.ai_player = ai_player
        self.ai_delay_ms = ai_delay_ms
        self.next_ai_at = 0
        self.wave_ms = wave_ms
        self.frames: list[BoardSnapshot] = []
        self.frame_index = 0
        self.next_frame_at = 0
        self.message = "Click a highlighted cell. R resets. Esc quits."
        self.schedule_ai_if_needed(pygame.time.get_ticks())

    def reset(self) -> None:
        self.env.reset()
        self.frames = []
        self.frame_index = 0
        self.next_frame_at = 0
        self.message = "New game. Blue / P1 to move."
        self.schedule_ai_if_needed(pygame.time.get_ticks())

    def run(self) -> None:
        running = True
        while running:
            now = pygame.time.get_ticks()
            running = self.handle_events()
            self.advance_animation(now)
            self.maybe_run_ai(now)
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_SPACE:
                    self.frames = []
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
        return True

    def handle_click(self, pos: tuple[int, int]) -> None:
        if self.frames or self.env.get_winner() != 0:
            return
        if self.is_ai_turn():
            self.message = f"{PLAYER_NAMES[self.ai_player]} is controlled by the checkpoint."
            return
        if not self.board_rect.collidepoint(pos):
            return

        x = (pos[0] - self.board_rect.left) // self.cell_size
        y = (pos[1] - self.board_rect.top) // self.cell_size
        action = int(y * BOARD_SIZE + x)
        player = current_player(self.env)
        self.perform_move(action, player, f"{PLAYER_NAMES[player]}", x, y)

    def perform_move(self, action: int, player: int, actor_label: str, x: int | None = None, y: int | None = None) -> bool:
        legal = self.env.legal_actions(player)
        if not legal[action]:
            if x is None or y is None:
                y, x = divmod(action, BOARD_SIZE)
            self.message = f"Illegal move for {PLAYER_NAMES[player]} at ({x}, {y})."
            return False

        accepted = self.env.step(action, player)
        if not accepted:
            self.message = "Core rejected move. State was not changed."
            return False

        self.frames = wave_snapshots(self.env)
        self.frame_index = 0
        self.next_frame_at = pygame.time.get_ticks() + self.wave_ms
        winner = self.env.get_winner()
        if winner:
            self.message = f"{PLAYER_NAMES[winner]} wins. Press R for the next explosion garden."
        elif self.frames:
            self.message = f"{actor_label} moved. Cascade resolved in {len(self.frames)} wave(s)."
        else:
            self.message = f"{actor_label} moved. {PLAYER_NAMES[current_player(self.env)]} to move."
        self.schedule_ai_if_needed(pygame.time.get_ticks())
        return True

    def is_ai_turn(self) -> bool:
        return self.policy is not None and current_player(self.env) == self.ai_player and self.env.get_winner() == 0

    def schedule_ai_if_needed(self, now: int) -> None:
        if self.is_ai_turn() and not self.frames:
            self.next_ai_at = now + self.ai_delay_ms
        else:
            self.next_ai_at = 0

    def maybe_run_ai(self, now: int) -> None:
        if not self.is_ai_turn() or self.frames:
            return
        if self.next_ai_at == 0:
            self.next_ai_at = now + self.ai_delay_ms
            return
        if now < self.next_ai_at:
            return
        assert self.policy is not None
        player = current_player(self.env)
        action = self.policy.select_action(self.env, player)
        y, x = divmod(action, BOARD_SIZE)
        self.perform_move(action, player, f"AI {PLAYER_NAMES[player]} ({x}, {y})", x, y)

    def advance_animation(self, now: int) -> None:
        if not self.frames:
            return
        if now < self.next_frame_at:
            return
        self.frame_index += 1
        self.next_frame_at = now + self.wave_ms
        if self.frame_index >= len(self.frames):
            self.frames = []
            self.frame_index = 0
            self.schedule_ai_if_needed(now)

    def visible_snapshot(self) -> BoardSnapshot:
        if self.frames:
            return self.frames[min(self.frame_index, len(self.frames) - 1)]
        return make_snapshot(self.env)

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        snapshot = self.visible_snapshot()
        player = current_player(self.env)
        legal = self.env.legal_actions(player) if self.env.get_winner() == 0 and not self.frames else [0] * CELL_COUNT
        self.draw_board(snapshot, legal)
        self.draw_panel(snapshot)
        self.draw_status(snapshot)

    def draw_board(self, snapshot: BoardSnapshot, legal: list[int]) -> None:
        pygame.draw.rect(self.screen, GRID, self.board_rect.inflate(4, 4), border_radius=10)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                idx = y * BOARD_SIZE + x
                rect = pygame.Rect(
                    self.board_rect.left + x * self.cell_size,
                    self.board_rect.top + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ).inflate(-6, -6)
                owner = int(snapshot.owners[idx])
                tokens = int(snapshot.tokens[idx])
                color = PLAYER_COLORS.get(owner, EMPTY)
                pygame.draw.rect(self.screen, color, rect, border_radius=12)

                if legal[idx]:
                    pygame.draw.rect(self.screen, LEGAL, rect, width=3, border_radius=12)
                elif owner:
                    pygame.draw.rect(self.screen, ILLEGAL, rect, width=1, border_radius=12)

                if snapshot.exploded[idx]:
                    pygame.draw.rect(self.screen, (255, 220, 98), rect, width=5, border_radius=12)

                if tokens > 0:
                    text = self.big_font.render(str(tokens), True, TEXT)
                    self.screen.blit(text, text.get_rect(center=rect.center))

    def draw_panel(self, snapshot: BoardSnapshot) -> None:
        panel_rect = pygame.Rect(
            self.board_rect.right + GRID_MARGIN,
            GRID_MARGIN,
            PANEL_WIDTH,
            self.board_px,
        )
        pygame.draw.rect(self.screen, PANEL, panel_rect, border_radius=12)

        winner = self.env.get_winner()
        player = current_player(self.env)
        lines = [
            "Chain Reaction",
            "",
            f"Turn: {self.env.turn_count}",
            f"To move: {PLAYER_NAMES[player] if not winner else '—'}",
            f"Winner: {PLAYER_NAMES[winner] if winner else 'none'}",
            f"Alive mask: {self.env.players_alive_mask:02b}",
            f"Last exploded: {bool(self.env.last_move_exploded)}",
            f"Waves: {self.env.wave_count}",
            f"Log truncated: {bool(self.env.wave_log_truncated)}",
        ]
        if self.policy is not None:
            lines.extend([
                "",
                f"AI: {PLAYER_NAMES[self.ai_player]}",
                f"Temp: {self.policy.temperature}",
            ])
        if snapshot.label:
            lines.extend(["", f"Showing {snapshot.label}"])

        y = panel_rect.top + 20
        for i, line in enumerate(lines):
            font = self.font if i == 0 else self.small_font
            color = TEXT if line else MUTED
            rendered = font.render(line, True, color)
            self.screen.blit(rendered, (panel_rect.left + 18, y))
            y += 30 if i == 0 else 24

        help_lines = [
            "Controls",
            "click: human move",
            "space: skip animation",
            "r: reset",
            "esc: quit",
        ]
        y = panel_rect.bottom - 142
        for i, line in enumerate(help_lines):
            font = self.font if i == 0 else self.small_font
            rendered = font.render(line, True, TEXT if i == 0 else MUTED)
            self.screen.blit(rendered, (panel_rect.left + 18, y))
            y += 25

    def draw_status(self, snapshot: BoardSnapshot) -> None:
        status_rect = pygame.Rect(
            GRID_MARGIN,
            self.board_rect.bottom + 18,
            self.width - GRID_MARGIN * 2,
            STATUS_HEIGHT - 22,
        )
        pygame.draw.rect(self.screen, PANEL, status_rect, border_radius=10)
        message = snapshot.label or self.message
        rendered = self.font.render(message, True, TEXT)
        self.screen.blit(rendered, (status_rect.left + 16, status_rect.top + 14))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Chain Reaction through pygame-ce.")
    parser.add_argument("--cell-size", type=int, default=72, help="Rendered cell size in pixels.")
    parser.add_argument("--wave-ms", type=int, default=220, help="Milliseconds per cascade wave frame.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Torch PPO checkpoint path, or 'latest' for the newest .pt under training/checkpoints.",
    )
    parser.add_argument("--ai-player", type=int, choices=(1, 2), default=2, help="Player controlled by the checkpoint.")
    parser.add_argument("--temperature", type=float, default=0.0, help="AI sampling temperature; 0 chooses argmax.")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint inference.")
    parser.add_argument("--ai-delay-ms", type=int, default=300, help="Delay before AI moves, for readability.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    checkpoint = resolve_checkpoint(args.checkpoint)
    policy = TorchPolicy(checkpoint, args.temperature, args.device) if checkpoint is not None else None
    if policy is not None:
        print(f"Loaded checkpoint AI: {policy.label} from {policy.checkpoint_path}", flush=True)
    app = ChainReactionApp(
        cell_size=args.cell_size,
        wave_ms=args.wave_ms,
        policy=policy,
        ai_player=args.ai_player,
        ai_delay_ms=args.ai_delay_ms,
    )
    app.run()


if __name__ == "__main__":
    main()
