import chain_reaction_core


def main():
    env = chain_reaction_core.PyChainReaction()

    assert env.get_winner() == 0
    assert sum(env.legal_actions(1)) == 64
    assert env.step(0, 1) == 1
    assert env.tokens[0] == 1
    assert env.owners[0] == 1
    assert env.turn_count == 1
    assert env.players_alive_mask == 3
    assert env.last_move_exploded == 0
    assert env.wave_count == 0
    assert env.wave_log_truncated == 0
    assert env.get_winner() == 0
    assert env.legal_actions(2)[0] == 0
    assert env.observation(1)[0] == 1
    assert env.observation(2)[0] == -1
    assert env.step(0, 2) == 0
    assert env.tokens[0] == 1
    assert env.owners[0] == 1

    env.reset()
    assert env.step(1, 1) == 1
    assert env.observation(1)[1] == 2
    assert env.observation(2)[1] == -2

    env.reset()
    assert env.step(9, 1) == 1
    assert env.observation(1)[9] == 3
    assert env.observation(2)[9] == -3

    assert env.step_fast(1, 2) == 1
    assert env.tokens[1] == 1
    assert env.owners[1] == 2
    assert env.wave_count == 0

    env.reset()
    assert env.step(0, 1) == 1
    assert env.step(0, 1) == 1
    assert env.wave_count == 1
    assert env.wave_log_truncated == 0
    assert env.wave_exploded[0] == 1
    assert env.wave_tokens[0] == 0
    assert env.wave_tokens[1] == 1
    assert env.wave_tokens[8] == 1

    env.reset()
    assert env.turn_count == 0
    assert env.players_alive_mask == 3
    assert env.last_move_exploded == 0
    assert env.wave_count == 0
    assert sum(env.tokens) == 0
    assert sum(env.owners) == 0

    print("Cython bridge smoke test passed.")


if __name__ == "__main__":
    main()
