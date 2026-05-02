import numpy as np

try:
    import gymnasium
    import pufferlib
    import pufferlib.emulation  # noqa: F401 - registers PufferEnv on pufferlib
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "training.puffer_env requires PufferLib v4 and gymnasium. "
        "Install the PufferLib toolchain before constructing this wrapper."
    ) from exc

try:
    from . import chain_reaction_core
except ImportError:  # Allows running from training/ after setup.py build_ext --inplace.
    import chain_reaction_core


BOARD_CELLS = 64
PLAYER_ONE = 1
PLAYER_TWO = 2
DEFAULT_MAX_TURNS = 4096


class ChainReactionEnv(pufferlib.PufferEnv):
    """Thin PufferLib-facing wrapper over the Cython C++ core.

    The core has no draw or max-turn rule. ``max_turns`` is a harness-level
    truncation cap so vectorized rollouts stay finite without changing game
    physics.
    """

    def __init__(self, max_turns=DEFAULT_MAX_TURNS, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(BOARD_CELLS,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Discrete(BOARD_CELLS)
        self.num_agents = 1
        super().__init__(buf)

        self.core = chain_reaction_core.PyChainReaction()
        self.current_player = PLAYER_ONE
        self.max_turns = int(max_turns)
        self.episode_steps = 0
        self._seed = seed

    def reset(self, seed=0):
        self.core.reset()
        self.current_player = PLAYER_ONE
        self.episode_steps = 0
        self.rewards[:] = 0.0
        self.terminals[:] = 0.0
        self.truncations[:] = 0.0
        self._write_observation()
        return self.observations, [{}]

    def step(self, actions):
        action = int(np.asarray(actions).reshape(-1)[0])

        self.rewards[:] = 0.0
        self.terminals[:] = 0.0
        self.truncations[:] = 0.0

        if action < 0 or action >= BOARD_CELLS or not self._is_legal(action):
            self.rewards[0] = -1.0
            self.terminals[0] = 1.0
            self._write_observation()
            return self.observations, self.rewards, self.terminals, self.truncations, [
                {"illegal_action": action}
            ]

        ok = self.core.step_fast(action, self.current_player)
        if ok != 1:
            self.rewards[0] = -1.0
            self.terminals[0] = 1.0
            self._write_observation()
            return self.observations, self.rewards, self.terminals, self.truncations, [
                {"rejected_action": action}
            ]

        self.episode_steps += 1
        winner = self.core.get_winner()
        if winner != 0:
            self.rewards[0] = 1.0 if winner == self.current_player else -1.0
            self.terminals[0] = 1.0
            self._write_observation()
            return self.observations, self.rewards, self.terminals, self.truncations, [
                {"winner": int(winner)}
            ]

        if self.episode_steps >= self.max_turns:
            self.truncations[0] = 1.0
            self._write_observation()
            return self.observations, self.rewards, self.terminals, self.truncations, [
                {"max_turns": self.max_turns}
            ]

        self.current_player = PLAYER_TWO if self.current_player == PLAYER_ONE else PLAYER_ONE
        self._write_observation()
        return self.observations, self.rewards, self.terminals, self.truncations, [{}]

    def legal_action_mask(self):
        return np.asarray(self.core.legal_actions(self.current_player), dtype=bool)

    def _is_legal(self, action):
        return self.core.legal_actions(self.current_player)[action] == 1

    def _write_observation(self):
        self.observations[0, :] = np.asarray(
            self.core.observation(self.current_player),
            dtype=np.float32,
        )

    def close(self):
        pass
