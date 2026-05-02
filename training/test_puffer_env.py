import importlib.util


def _puffer_available():
    return importlib.util.find_spec("pufferlib") is not None and importlib.util.find_spec("gymnasium") is not None


def main():
    if not _puffer_available():
        print("PufferLib smoke test skipped: pufferlib/gymnasium not installed.")
        return

    import numpy as np
    from training.puffer_env import ChainReactionEnv

    env = ChainReactionEnv(max_turns=2)
    observations, infos = env.reset()
    assert observations.shape == (1, 64)
    assert observations.dtype == np.float32
    assert observations.min() >= -3.0
    assert observations.max() <= 3.0
    assert env.legal_action_mask().sum() == 64

    observations, rewards, terminals, truncations, infos = env.step([0])
    assert rewards[0] == 0.0
    assert terminals[0] == 0.0
    assert truncations[0] == 0.0
    assert env.current_player == 2
    assert env.legal_action_mask()[0] == 0
    assert observations[0, 0] == 1.0

    observations, rewards, terminals, truncations, infos = env.step([0])
    assert rewards[0] == -1.0
    assert terminals[0] == 1.0
    assert truncations[0] == 0.0
    assert infos[0]["illegal_action"] == 0

    env.reset()
    env.step([0])
    observations, rewards, terminals, truncations, infos = env.step([1])
    assert terminals[0] == 0.0
    assert truncations[0] == 1.0
    assert infos[0]["max_turns"] == 2

    env.close()

    import pufferlib.vector

    vecenv = pufferlib.vector.make(
        ChainReactionEnv,
        num_envs=2,
        backend=pufferlib.vector.Serial,
        env_kwargs={"max_turns": 4},
    )
    observations, infos = vecenv.reset()
    assert observations.shape == (2, 64)
    actions = np.asarray([0, 1])
    observations, rewards, terminals, truncations, infos = vecenv.step(actions)
    assert observations.shape == (2, 64)
    assert rewards.shape == (2,)
    assert terminals.shape == (2,)
    assert truncations.shape == (2,)
    vecenv.close()

    print("PufferLib wrapper smoke test passed.")


if __name__ == "__main__":
    main()
