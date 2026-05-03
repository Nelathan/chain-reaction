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
    # At t=1, terminal delta = 1 - 0.5. At t=0, the next return is
    # opponent-perspective and therefore subtracts from the current perspective:
    # G0 = 0 - G1 = -1, A0 = G0 - V0 = -1.25.
    assert torch.allclose(advantages[:, 0], torch.tensor([-1.25, 0.5]))
    assert returns.shape == rewards.shape

    bootstrap_rewards = torch.tensor([[0.0], [0.0]])
    bootstrap_values = torch.zeros(2, 1)
    bootstrap_terminals = torch.zeros(2, 1)
    bootstrap_advantages, bootstrap_returns = compute_negamax_gae(
        bootstrap_rewards,
        bootstrap_values,
        bootstrap_terminals,
        torch.tensor([0.7]),
        1.0,
        1.0,
    )
    # With no rewards and no terminal, a good position for the player after the
    # rollout is bad for the player at the previous ply, then good again one ply
    # earlier: G1 = -0.7, G0 = 0.7.
    assert torch.allclose(bootstrap_advantages[:, 0], torch.tensor([0.7, -0.7]))
    assert torch.allclose(bootstrap_returns[:, 0], torch.tensor([0.7, -0.7]))

    batch_rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    batch_values = torch.zeros(2, 2)
    batch_terminals = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    batch_advantages, _ = compute_negamax_gae(
        batch_rewards,
        batch_values,
        batch_terminals,
        torch.tensor([0.9, 0.5]),
        1.0,
        1.0,
    )
    # Env 0 carries the bootstrap backward with alternating signs. Env 1
    # terminates at t=0, so t=0 must not leak into the reset episode at t=1.
    assert torch.allclose(batch_advantages[:, 0], torch.tensor([0.9, -0.9]))
    assert torch.allclose(batch_advantages[:, 1], torch.tensor([1.0, -0.5]))

    print("Torch PPO module smoke test passed.")


if __name__ == "__main__":
    main()
