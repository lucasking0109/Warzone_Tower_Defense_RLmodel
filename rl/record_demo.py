"""
Record human gameplay for imitation learning.

Usage:
    python3 rl/record_demo.py [--seed 42] [--output demos/demo.npz]

Controls:
    1-7     Select tower (1=MG 2=CN 3=FR 4=SK 5=LS 6=AT 7=PL)
    Q       Select wall mode
    U       Toggle upgrade mode (click on tower to upgrade)
    Click   Place tower/wall or upgrade tower
    SPACE   Pause / Resume
    +/-     Speed up / slow down
    ESC     Quit & save
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from rl.td_env import TowerDefenseEnv, TOWER_TYPE_LIST
from simulator.renderer import Renderer
from simulator.game_config import TILE_SIZE, TOWER_TYPES, WALL_BLOCK_COST


def find_tower_at(env, gx, gy):
    """Find tower slot that covers grid position (gx, gy). Towers are 2x2."""
    for slot_idx, tower in env._slot_towers.items():
        tx, ty = tower.grid_x, tower.grid_y
        if tx <= gx <= tx + 1 and ty <= gy <= ty + 1:
            return slot_idx, tower
    return None, None


def find_wall_pos_idx(env, gx, gy):
    """Find wall position index for grid position (gx, gy)."""
    pos = (gx, gy)
    for i, wp in enumerate(env._wall_positions):
        if wp == pos:
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="Record gameplay for imitation learning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="demos/demo.npz")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env = TowerDefenseEnv()
    obs, info = env.reset(seed=args.seed)
    renderer = Renderer()

    print("=" * 50)
    print("  RECORDING MODE — Imitation Learning")
    print("=" * 50)
    print()
    print("Controls:")
    print("  1-7     Select tower")
    print("          1=MG($250) 2=CN($400) 3=FR($600)")
    print("          4=SK($800) 5=LS($800) 6=AT($1000) 7=PL($1000)")
    print("  Q       Select wall ($50)")
    print("  U       Toggle upgrade mode")
    print("  Click   Place / Upgrade")
    print("  SPACE   Pause / Resume")
    print("  +/-     Speed up / slow down")
    print("  ESC     Quit & save")
    print()
    print(f"Action space: {env.ACTION_SPACE_N}")
    print(f"Obs space: {env.OBS_SIZE}")
    print(f"Seed: {args.seed}")
    print()

    tool_keys = {
        pygame.K_1: "bulletTower",
        pygame.K_2: "cannonTower",
        pygame.K_3: "fireTower",
        pygame.K_4: "shockTower",
        pygame.K_5: "laserTower",
        pygame.K_6: "artilleryTower",
        pygame.K_7: "PlasmaTower",
    }

    TOWER_ABBREVS = {
        "bulletTower": "MG", "cannonTower": "CN", "fireTower": "FR",
        "shockTower": "SK", "laserTower": "LS", "artilleryTower": "AT",
        "PlasmaTower": "PL",
    }

    observations = []
    actions_list = []

    selected_type = None  # tower type name or "wall"
    upgrade_mode = False
    paused = False
    game_over = False

    # Speed: frames between auto-NOOP steps
    # At 20 FPS, step_delay=20 → 1 step/sec → game runs at 1.5x
    step_delay_frames = 20
    frame_counter = 0
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("⏸  PAUSED" if paused else "▶  RESUMED")

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    step_delay_frames = max(3, step_delay_frames - 3)
                    speed = 20.0 / step_delay_frames * 1.5
                    print(f"Speed: {speed:.1f}x")

                elif event.key == pygame.K_MINUS:
                    step_delay_frames = min(60, step_delay_frames + 3)
                    speed = 20.0 / step_delay_frames * 1.5
                    print(f"Speed: {speed:.1f}x")

                elif event.key == pygame.K_u:
                    upgrade_mode = not upgrade_mode
                    if upgrade_mode:
                        selected_type = None
                        print(">> UPGRADE MODE (click on tower)")
                    else:
                        print(">> BUILD MODE")


                elif event.key == pygame.K_q:
                    upgrade_mode = False
                    selected_type = "wall"
                    print(f"Selected: Wall (${WALL_BLOCK_COST})")

                elif event.key in tool_keys:
                    upgrade_mode = False
                    selected_type = tool_keys[event.key]
                    name = TOWER_TYPES[selected_type]["name"]
                    cost = TOWER_TYPES[selected_type]["levels"][1]["cost"]
                    abbr = TOWER_ABBREVS[selected_type]
                    print(f"Selected: {abbr} {name} (${cost})")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if paused or game_over:
                    continue

                mx, my = event.pos
                gx = mx // TILE_SIZE
                gy = my // TILE_SIZE

                if upgrade_mode:
                    # --- UPGRADE ---
                    slot_idx, tower = find_tower_at(env, gx, gy)
                    if tower is not None:
                        action = env.ACTION_UPGRADE_START + slot_idx
                        masks = env.action_masks()
                        if masks[action]:
                            observations.append(obs.copy())
                            actions_list.append(action)
                            obs, reward, term, trunc, info = env.step(action)
                            abbr = TOWER_ABBREVS.get(tower.tower_type, "??")
                            print(f"  [{len(observations)}] Upgraded {abbr} at ({tower.grid_x},{tower.grid_y}) "
                                  f"to Lv{tower.level} | Wave {info['wave']} | ${info['cash']}")
                            frame_counter = 0
                            if term or trunc:
                                print(f"\n=== GAME OVER! Waves: {info['wave']} | Steps: {len(observations)} ===")
                                game_over = True
                        else:
                            print(f"  Can't upgrade (max level or no cash)")
                    else:
                        print(f"  No tower at ({gx},{gy})")

                elif selected_type == "wall":
                    # --- WALL ---
                    wall_idx = find_wall_pos_idx(env, gx, gy)
                    if wall_idx is not None:
                        action = env.ACTION_WALL_START + wall_idx
                        masks = env.action_masks()
                        if masks[action]:
                            observations.append(obs.copy())
                            actions_list.append(action)
                            obs, reward, term, trunc, info = env.step(action)
                            print(f"  [{len(observations)}] Wall at ({gx},{gy}) | "
                                  f"Wave {info['wave']} | ${info['cash']} | Walls {info['wall_count']}")
                            frame_counter = 0
                            if term or trunc:
                                print(f"\n=== GAME OVER! Waves: {info['wave']} | Steps: {len(observations)} ===")
                                game_over = True
                        else:
                            print(f"  Can't place wall at ({gx},{gy}) (blocked or no cash)")
                    else:
                        print(f"  Invalid wall position ({gx},{gy})")

                elif selected_type:
                    # --- BUILD TOWER ---
                    type_idx = TOWER_TYPE_LIST.index(selected_type)
                    pos = (gx, gy)
                    if pos in env._pos_to_idx:
                        pos_idx = env._pos_to_idx[pos]
                        action = env.ACTION_BUILD_START + type_idx * env.N_POS + pos_idx
                        masks = env.action_masks()
                        if masks[action]:
                            observations.append(obs.copy())
                            actions_list.append(action)
                            obs, reward, term, trunc, info = env.step(action)
                            abbr = TOWER_ABBREVS[selected_type]
                            print(f"  [{len(observations)}] Built {abbr} at ({gx},{gy}) | "
                                  f"Wave {info['wave']} | ${info['cash']} | Towers {info['num_towers']}")
                            frame_counter = 0
                            if term or trunc:
                                print(f"\n=== GAME OVER! Waves: {info['wave']} | Steps: {len(observations)} ===")
                                game_over = True
                        else:
                            print(f"  Can't build at ({gx},{gy}) (blocked or no cash)")
                    else:
                        print(f"  Invalid tower position ({gx},{gy})")

        # --- Auto-advance with NOOP ---
        if not paused and not game_over and running:
            frame_counter += 1
            if frame_counter >= step_delay_frames:
                frame_counter = 0
                observations.append(obs.copy())
                actions_list.append(env.ACTION_NOOP)
                obs, reward, term, trunc, info = env.step(env.ACTION_NOOP)
                if term or trunc:
                    wn = info.get('wave', 0)
                    print(f"\n=== GAME OVER! Waves: {wn} | Steps: {len(observations)} ===")
                    game_over = True

        # --- Render ---
        if not renderer.render(env.engine):
            running = False

        clock.tick(20)  # 20 FPS

    # --- Save recording ---
    if observations:
        np.savez_compressed(
            args.output,
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions_list, dtype=np.int64),
            seed=args.seed,
        )
        n_noop = sum(1 for a in actions_list if a == 0)
        n_build = sum(1 for a in actions_list
                      if env.ACTION_BUILD_START <= a <= env.ACTION_BUILD_END)
        n_upgrade = sum(1 for a in actions_list
                        if env.ACTION_UPGRADE_START <= a <= env.ACTION_UPGRADE_END)
        n_wall = sum(1 for a in actions_list
                     if env.ACTION_WALL_START <= a <= env.ACTION_WALL_END)
        print(f"\nSaved {len(observations)} steps to {args.output}")
        print(f"  NOOP: {n_noop} | BUILD: {n_build} | UPGRADE: {n_upgrade} | WALL: {n_wall}")
    else:
        print("\nNo data recorded.")

    renderer.close()


if __name__ == "__main__":
    main()
