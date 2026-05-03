from __future__ import annotations

import torch

from training.torch_ppo.gae import compute_negamax_gae
from training.torch_ppo.masking import apply_legal_mask, legal_action_mask
from training.torch_ppo.model import ChainReactionNet


def main() -> None:
    obs = torch.zeros(3, 64)
    obs[0, 0] = -1
    obs[1, 7] = -2
    obs[2, 63] = 1

    model = ChainReactionNet()
    logits, values = model(obs)
    assert logits.shape == (3, 64)
    assert values.shape == (3,)

    mask = legal_action_mask(obs)
    assert not mask[0, 0]
    assert mask[0, 1]

    masked = apply_legal_mask(logits, obs)
    assert masked[0, 0] < -1e30

    rewards = torch.tensor([[0.0], [1.0]])
    vals = torch.tensor([[0.25], [0.50]])
    terminals = torch.tensor([[0.0], [1.0]])
    advantages, returns = compute_negamax_gae(rewards, vals, terminals, torch.tensor([0.0]), 1.0, 1.0)
    # At t=1, terminal delta = 1 - 0.5. At t=0, the next advantage/value are
    # opponent-perspective and therefore subtract from the current perspective.
    assert torch.allclose(advantages[:, 0], torch.tensor([-1.25, 0.5]))
    assert returns.shape == rewards.shape

    print("Torch PPO module smoke test passed.")


if __name__ == "__main__":
    main()
