"""
Warzone TD — Visual Replay of a trained RL agent.
Uses Pygame renderer to show the agent's gameplay step-by-step.
Uses env.step() directly for correctness (matches training/eval exactly).

Usage:
    python rl/replay.py                         # replay final model
    python rl/replay.py --model best_model/best_model
    python rl/replay.py --speed 5               # 5 steps per second
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.distributions.Distribution.set_default_validate_args(False)

import pygame
from sb3_contrib import MaskablePPO

from rl.td_env import TowerDefenseEnv, TOWER_TYPE_LIST, MAX_WALLS
from simulator.renderer import Renderer, SCREEN_WIDTH, SCREEN_HEIGHT
from simulator.game_config import GRID_HEIGHT, TILE_SIZE, TOWER_TYPES

# Tower type short names for display
TOWER_SHORT_NAMES = {
    "bulletTower": "MG",
    "cannonTower": "CN",
    "fireTower": "FR",
    "shockTower": "SK",
    "laserTower": "LS",
    "artilleryTower": "AT",
    "PlasmaTower": "PL",
}


def replay(args):
    # --- Load model ---
    model = MaskablePPO.load(args.model)
    print(f"Loaded model: {args.model}")

    # --- Create env ---
    env = TowerDefenseEnv()
    obs, info = env.reset(seed=args.seed)
    print(f"Seed: {args.seed}")
    engine = env.engine

    # --- Pygame renderer ---
    renderer = Renderer()
    font = pygame.font.SysFont("monospace", 15, bold=True)
    font_small = pygame.font.SysFont("monospace", 13)

    speed = args.speed
    paused = False
    step_count = 0
    total_reward = 0.0
    last_reward = 0.0
    last_action_text = "Waiting..."
    done = False

    print("Controls:")
    print("  SPACE  = Pause / Resume")
    print("  +/-    = Speed up / slow down")
    print("  ESC    = Quit")
    print()

    running = True
    while running:
        # --- Handle input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(60, speed + 1)
                    print(f"Speed: {speed} steps/sec")
                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 1)
                    print(f"Speed: {speed} steps/sec")

        if not running:
            break

        # --- Paused or done: just render, don't advance ---
        if paused or done:
            renderer.render(engine)
            _draw_overlay(renderer.screen, font, font_small, engine, env,
                          step_count, last_action_text, total_reward, last_reward,
                          paused, speed, done)
            pygame.display.flip()
            renderer.clock.tick(30)

            if done:
                # Wait for user to close
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                            running = False
            continue

        # === Agent step (uses env.step() for correctness) ===
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        # Decode action for display
        last_action_text = _decode_action(env, int(action))

        # Execute full RL step
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        last_reward = reward
        step_count += 1
        done = terminated or truncated

        if done:
            print(f"\n=== Episode Complete ===")
            print(f"Waves survived: {info.get('wave', 0)}")
            print(f"Total kills: {info.get('episode_kills', 0)}")
            print(f"Total leaks: {info.get('episode_leaks', 0)}")
            print(f"Score: {info.get('score', 0)}")
            print(f"Total reward: {total_reward:.1f}")
            print(f"Tower composition: {_tower_composition(engine)}")

        # --- Render ---
        renderer.render(engine)
        _draw_overlay(renderer.screen, font, font_small, engine, env,
                      step_count, last_action_text, total_reward, last_reward,
                      paused, speed, done)
        pygame.display.flip()

        # Speed control: N steps per second
        renderer.clock.tick(speed)

    renderer.close()


def _tower_composition(engine):
    """Return a string like 'MG×8 CN×5 FR×3'."""
    counts = Counter()
    for tower in engine.towers:
        short = TOWER_SHORT_NAMES.get(tower.tower_type, tower.tower_type[:2])
        counts[short] += 1
    if not counts:
        return "none"
    return " ".join(f"{name}×{cnt}" for name, cnt in counts.most_common())


def _draw_overlay(screen, font, font_small, engine, env,
                  step_count, action_text, total_reward, last_reward,
                  paused, speed, done):
    """Draw an info overlay on top of the rendered game."""
    hud_y = GRID_HEIGHT * TILE_SIZE + 35

    # Tower composition
    comp = _tower_composition(engine)

    lines = [
        f"Step {step_count}  |  Action: {action_text}  |  "
        f"Reward: {total_reward:+.1f}  (step {last_reward:+.2f})  |  "
        f"Speed: {speed}" + ("  [PAUSED]" if paused else ""),
        f"Towers: {len(engine.towers)} [{comp}]  |  "
        f"Walls: {env._wall_count}  |  "
        f"Kills: {env._episode_kills}  |  Leaks: {env._episode_leaks}",
    ]

    for i, line in enumerate(lines):
        surf = font_small.render(line, True, (200, 200, 200))
        screen.blit(surf, (10, hud_y + i * 14))

    if done:
        banner = font.render(
            f"DONE — Wave {engine.wave_controller.wave_number} | "
            f"Kills {env._episode_kills} | Leaks {env._episode_leaks} | "
            f"Reward {total_reward:+.1f} | "
            f"Press ESC/SPACE to exit",
            True, (255, 255, 100)
        )
        bx = SCREEN_WIDTH // 2 - banner.get_width() // 2
        screen.blit(banner, (bx, SCREEN_HEIGHT // 2))


def _decode_action(env, action):
    """Convert action index to human-readable string."""
    if action == env.ACTION_NOOP:
        return "NOOP"
    elif env.ACTION_BUILD_START <= action <= env.ACTION_BUILD_END:
        build_idx = action - env.ACTION_BUILD_START
        type_idx = build_idx // env.N_POS
        pos_idx = build_idx % env.N_POS
        tower_type = TOWER_TYPE_LIST[type_idx]
        x, y = env._candidate_positions[pos_idx]
        name = TOWER_TYPES[tower_type]["name"]
        short = TOWER_SHORT_NAMES.get(tower_type, "??")
        return f"BUILD {name}({short}) @ ({x},{y})"
    elif env.ACTION_UPGRADE_START <= action <= env.ACTION_UPGRADE_END:
        slot_idx = action - env.ACTION_UPGRADE_START
        tower = env._slot_towers.get(slot_idx)
        if tower:
            name = TOWER_TYPES[tower.tower_type]["name"]
            short = TOWER_SHORT_NAMES.get(tower.tower_type, "??")
            return f"UPGRADE {name}({short}) Lv{tower.level} @ ({tower.grid_x},{tower.grid_y})"
        return f"UPGRADE slot {slot_idx} (empty)"
    elif env.ACTION_WALL_START <= action <= env.ACTION_WALL_END:
        pos_idx = action - env.ACTION_WALL_START
        x, y = env._wall_positions[pos_idx]
        return f"WALL @ ({x},{y})"
    return f"UNKNOWN({action})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual replay of trained agent")
    parser.add_argument("--model", type=str, default="warzone_td_final",
                        help="Path to model (default: warzone_td_final)")
    parser.add_argument("--speed", type=int, default=5,
                        help="Steps per second (default: 5, max: 60)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    replay(args)
