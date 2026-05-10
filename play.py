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


class ChainReactionApp:
    def __init__(self, cell_size: int, wave_ms: int) -> None:
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
        self.wave_ms = wave_ms
        self.frames: list[BoardSnapshot] = []
        self.frame_index = 0
        self.next_frame_at = 0
        self.message = "Click a highlighted cell. R resets. Esc quits."

    def reset(self) -> None:
        self.env.reset()
        self.frames = []
        self.frame_index = 0
        self.next_frame_at = 0
        self.message = "New game. Blue / P1 to move."

    def run(self) -> None:
        running = True
        while running:
            now = pygame.time.get_ticks()
            running = self.handle_events()
            self.advance_animation(now)
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
        if not self.board_rect.collidepoint(pos):
            return

        x = (pos[0] - self.board_rect.left) // self.cell_size
        y = (pos[1] - self.board_rect.top) // self.cell_size
        action = int(y * BOARD_SIZE + x)
        player = current_player(self.env)
        legal = self.env.legal_actions(player)
        if not legal[action]:
            self.message = f"Illegal move for {PLAYER_NAMES[player]} at ({x}, {y})."
            return

        accepted = self.env.step(action, player)
        if not accepted:
            self.message = "Core rejected move. State was not changed."
            return

        self.frames = wave_snapshots(self.env)
        self.frame_index = 0
        self.next_frame_at = pygame.time.get_ticks() + self.wave_ms
        winner = self.env.get_winner()
        if winner:
            self.message = f"{PLAYER_NAMES[winner]} wins. Press R for the next explosion garden."
        elif self.frames:
            self.message = f"Cascade resolved in {len(self.frames)} wave(s)."
        else:
            self.message = f"Accepted. {PLAYER_NAMES[current_player(self.env)]} to move."

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
            "click: core step",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = ChainReactionApp(cell_size=args.cell_size, wave_ms=args.wave_ms)
    app.run()


if __name__ == "__main__":
    main()
