import chain_reaction_core


def main():
    env = chain_reaction_core.PyChainReaction()

    assert env.get_winner() == 0
    assert sum(env.legal_actions(1)) == 64
    assert env.step(0, 1) == 1
    assert env.tokens[0] == 1
    assert env.owners[0] == 1
    assert env.turn_count == 1
    assert env.players_seen_mask == 1
    assert env.players_alive_mask == 1
    assert env.last_move_exploded == 0
    assert env.get_winner() == 0
    assert env.legal_actions(2)[0] == 0
    assert env.observation(1)[0] == -1
    assert env.observation(2)[0] == 1
    assert env.step(0, 2) == 0
    assert env.tokens[0] == 1
    assert env.owners[0] == 1

    env.reset()
    assert env.turn_count == 0
    assert env.players_seen_mask == 0
    assert env.players_alive_mask == 0
    assert env.last_move_exploded == 0
    assert sum(env.tokens) == 0
    assert sum(env.owners) == 0

    print("Cython bridge smoke test passed.")


if __name__ == "__main__":
    main()
