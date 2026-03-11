"""
Warzone Tower Defense — RL Training Script
Uses MaskablePPO from sb3-contrib with action masking.
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix PyTorch 2.6+ simplex validation issue with large discrete action spaces
import torch
torch.distributions.Distribution.set_default_validate_args(False)

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.td_env import TowerDefenseEnv


class WaveBestModelCallback(BaseCallback):
    """Save best model based on average wave count (not reward)."""

    def __init__(self, eval_env, save_path, eval_freq=2500,
                 n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_waves = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        waves = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                masks = get_action_masks(self.eval_env)
                action, _ = self.model.predict(obs, deterministic=True,
                                               action_masks=masks)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            waves.append(info.get("wave", 0))

        mean_waves = np.mean(waves)
        if self.verbose:
            print(f"Wave eval @ {self.num_timesteps}: "
                  f"mean={mean_waves:.1f}, max={max(waves)}, "
                  f"best={self.best_mean_waves:.1f}")

        if mean_waves > self.best_mean_waves:
            self.best_mean_waves = mean_waves
            path = os.path.join(self.save_path, "best_wave_model")
            self.model.save(path)
            if self.verbose:
                print(f"  New best wave model! ({mean_waves:.1f} waves) "
                      f"saved to {path}")

        return True

N_ENVS = 10  # parallel environments


def make_env():
    """Return a callable that creates a single env (for SubprocVecEnv)."""
    def _init():
        env = TowerDefenseEnv()
        env = Monitor(env)
        return env
    return _init


def make_single_env():
    """Create a single env (for eval)."""
    env = TowerDefenseEnv()
    env = Monitor(env)
    return env


def get_run_dir(run_name: str) -> str:
    """Get the output directory for a training run."""
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train(args):
    run_dir = get_run_dir(args.run_name)

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    eval_env = make_single_env()

    print(f"Run: {args.run_name} → {run_dir}/")
    print(f"Training with {N_ENVS} parallel environments")
    print(f"Action space: {env.action_space.n}")
    print(f"Observation space: {env.observation_space.shape}")

    # All outputs go under runs/<run_name>/
    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // N_ENVS, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="warzone_td",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=max(25_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    wave_eval_env = make_single_env()
    wave_cb = WaveBestModelCallback(
        wave_eval_env,
        save_path=run_dir,
        eval_freq=max(200_000 // N_ENVS, 1),
        n_eval_episodes=5,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb, wave_cb])

    net_size = args.net_size
    print(f"Hyperparams: ent_coef={args.ent_coef}, net=[{net_size},{net_size}]")

    if args.resume:
        print(f"Resuming from pretrained model: {args.resume}")
        model = MaskablePPO.load(
            args.resume,
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            tensorboard_log=os.path.join(run_dir, "tb_logs"),
            seed=42,
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(
                net_arch=dict(pi=[net_size, net_size], vf=[net_size, net_size]),
            ),
            tensorboard_log=os.path.join(run_dir, "tb_logs"),
            seed=42,
        )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(run_dir, "warzone_td_final")
    model.save(final_path)
    print(f"Training complete! Model saved to {final_path}.zip")


def evaluate(args):
    """Run a trained model and print stats."""
    env = make_single_env()

    model = MaskablePPO.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if info.get("action_type", "").endswith("_ok"):
            print(f"  Step {info['step_count']}: {info['action_type']} | "
                  f"Wave {info['wave']} | Cash ${info['cash']} | "
                  f"Towers {info['num_towers']} | HP {info['base_hp']:.1f}")

    print(f"\n=== Episode Complete ===")
    print(f"Waves survived: {info.get('wave', 0)}")
    print(f"Total kills: {info.get('episode_kills', 0)}")
    print(f"Total leaks: {info.get('episode_leaks', 0)}")
    print(f"Final score: {info.get('score', 0)}")
    print(f"Total reward: {total_reward:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warzone TD RL Training")
    sub = parser.add_subparsers(dest="command")

    # Train command
    train_p = sub.add_parser("train", help="Train a new model")
    train_p.add_argument("--timesteps", type=int, default=10_000_000)
    train_p.add_argument("--run-name", type=str, required=True,
                         help="Name for this run (e.g. run3_explore)")
    train_p.add_argument("--ent-coef", type=float, default=0.03,
                         help="Entropy coefficient (default: 0.03)")
    train_p.add_argument("--net-size", type=int, default=512,
                         help="Hidden layer size for pi/vf networks (default: 512)")
    train_p.add_argument("--resume", type=str, default=None,
                         help="Path to pretrained model to resume from (e.g. pretrained_model)")

    # Evaluate command
    eval_p = sub.add_parser("eval", help="Evaluate a trained model")
    eval_p.add_argument("--model-path", type=str, default="warzone_td_final")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()
