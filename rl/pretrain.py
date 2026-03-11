"""
Behavioral Cloning: pretrain RL agent from human demonstrations.

Usage:
    python3 rl/pretrain.py --demo demos/demo.npz --output pretrained_model
    python3 rl/pretrain.py --demo demos/*.npz --output pretrained_model  # multiple demos
    python3 rl/pretrain.py --demo demos/*.npz --resume runs/run16_balanced/best_wave_model --output pretrained_model
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from rl.td_env import TowerDefenseEnv


def get_action_logits(policy, obs_tensor):
    """Get raw action logits from policy network."""
    features = policy.features_extractor(obs_tensor)
    latent_pi, _ = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def main():
    parser = argparse.ArgumentParser(description="Pretrain agent via behavioral cloning")
    parser.add_argument("--demo", type=str, nargs="+", required=True,
                        help="Path(s) to demo .npz files")
    parser.add_argument("--output", type=str, default="pretrained_model",
                        help="Output model path (without .zip)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--net-size", type=int, default=512)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing model to fine-tune (e.g. runs/run16_balanced/best_wave_model)")
    args = parser.parse_args()

    # --- Load demos ---
    all_obs, all_act = [], []
    for path in args.demo:
        data = np.load(path)
        all_obs.append(data["observations"])
        all_act.append(data["actions"])
        print(f"Loaded {len(data['observations'])} steps from {path}")

    observations = np.concatenate(all_obs).astype(np.float32)
    actions = np.concatenate(all_act).astype(np.int64)
    print(f"Total: {len(observations)} steps")

    # Action distribution
    unique, counts = np.unique(actions, return_counts=True)
    print(f"\nAction distribution ({len(unique)} unique actions):")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        label = "NOOP" if u == 0 else f"action {u}"
        print(f"  {label}: {c} ({c/len(actions)*100:.1f}%)")

    # --- Create env and model ---
    env = TowerDefenseEnv()
    env = Monitor(env)

    if args.resume:
        print(f"\nFine-tuning from existing model: {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
    else:
        net_size = args.net_size
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=args.lr,
            policy_kwargs=dict(
                net_arch=dict(pi=[net_size, net_size], vf=[net_size, net_size]),
            ),
        )

    policy = model.policy
    device = policy.device
    print(f"\nDevice: {device}")
    if not args.resume:
        print(f"Network: pi=[{net_size},{net_size}], vf=[{net_size},{net_size}]")

    # --- Prepare data ---
    obs_tensor = torch.tensor(observations).to(device)
    act_tensor = torch.tensor(actions).to(device)
    dataset = TensorDataset(obs_tensor, act_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- Train ---
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nPretraining for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_obs, batch_act in dataloader:
            logits = get_action_logits(policy, batch_obs)
            loss = loss_fn(logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_obs)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_act).sum().item()
            total += len(batch_obs)

        avg_loss = total_loss / total
        acc = correct / total * 100

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(args.output)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            marker = " *" if avg_loss <= best_loss else ""
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"loss={avg_loss:.4f}, accuracy={acc:.1f}%{marker}")

    # Final save
    model.save(args.output)
    print(f"\nPretrained model saved to {args.output}.zip")
    print(f"Best loss: {best_loss:.4f}")
    print()
    print("Next step — continue training with RL:")
    print(f"  python3 rl/train.py train --run-name run17_imitation "
          f"--resume {args.output} --timesteps 10000000")

    env.close()


if __name__ == "__main__":
    main()
