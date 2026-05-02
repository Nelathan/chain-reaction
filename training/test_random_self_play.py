import random

import chain_reaction_core


def main():
    rng = random.Random(1234)
    max_steps = 2048

    for _episode in range(4):
        env = chain_reaction_core.PyChainReaction()
        player_id = 1

        for _step in range(max_steps):
            mask = env.legal_actions(player_id)
            legal_actions = [idx for idx, value in enumerate(mask) if value == 1]
            assert legal_actions

            action = rng.choice(legal_actions)
            assert env.step_fast(action, player_id) == 1

            observation = env.observation(1)
            assert min(observation) >= -3
            assert max(observation) <= 3

            if env.get_winner() != 0:
                break

            player_id = 2 if player_id == 1 else 1
        else:
            raise AssertionError("random legal self-play episode hit the smoke-test harness cap")

    print("Random legal self-play smoke test passed.")


if __name__ == "__main__":
    main()
