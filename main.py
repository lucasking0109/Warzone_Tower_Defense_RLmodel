"""
Warzone Tower Defense Simulator — Main entry point.
Run with: python main.py
"""
import sys
import pygame
from simulator.game_engine import GameEngine
from simulator.renderer import Renderer
from simulator.game_config import TILE_SIZE, TICK_MS, TOWER_TYPES, WALL_BLOCK_COST


def main():
    engine = GameEngine()
    renderer = Renderer()

    # Simulation speed
    ticks_per_frame = 1  # increase for fast-forward
    paused = False
    selected_tool = None  # tower type or "wall"
    show_help = True

    print("=== Warzone TD Simulator ===")
    print("Controls:")
    print("  1-7: Select tower (1=MG, 2=Cannon, 3=Flame, 4=Pulse, 5=Laser, 6=Heavy, 7=Plasma)")
    print("  Q: Select wall block")
    print("  Click: Place selected item")
    print("  SPACE: Pause/Resume")
    print("  +/-: Speed up/slow down")
    print("  ESC: Quit")
    print()

    tool_keys = {
        pygame.K_1: "bulletTower",
        pygame.K_2: "cannonTower",
        pygame.K_3: "fireTower",
        pygame.K_4: "shockTower",
        pygame.K_5: "laserTower",
        pygame.K_6: "artilleryTower",
        pygame.K_7: "PlasmaTower",
        pygame.K_q: "wall",
    }

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    ticks_per_frame = min(20, ticks_per_frame + 1)
                    print(f"Speed: {ticks_per_frame}x")
                elif event.key == pygame.K_MINUS:
                    ticks_per_frame = max(1, ticks_per_frame - 1)
                    print(f"Speed: {ticks_per_frame}x")
                elif event.key in tool_keys:
                    selected_tool = tool_keys[event.key]
                    if selected_tool == "wall":
                        print(f"Selected: Wall Block (${WALL_BLOCK_COST})")
                    else:
                        cost = TOWER_TYPES[selected_tool]["levels"][1]["cost"]
                        name = TOWER_TYPES[selected_tool]["name"]
                        print(f"Selected: {name} (${cost})")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                grid_x = mx // TILE_SIZE
                grid_y = my // TILE_SIZE

                if selected_tool == "wall":
                    if engine.place_wall(grid_x, grid_y):
                        print(f"Placed wall at ({grid_x}, {grid_y})")
                    else:
                        print(f"Can't place wall at ({grid_x}, {grid_y})")
                elif selected_tool and selected_tool in TOWER_TYPES:
                    if engine.build_tower(selected_tool, grid_x, grid_y):
                        name = TOWER_TYPES[selected_tool]["name"]
                        print(f"Built {name} at ({grid_x}, {grid_y})")
                    else:
                        print(f"Can't build tower at ({grid_x}, {grid_y})")

        # Update simulation
        if not paused:
            for _ in range(ticks_per_frame):
                info = engine.tick()
                if info.get("killed"):
                    for e in info["killed"]:
                        pass  # silent
                if info.get("leaked"):
                    for e in info["leaked"]:
                        print(f"  LEAKED: {e.creep_type} (dmg={e.damage_multiplier})")
                if info.get("game_over"):
                    print(f"\n=== GAME OVER === Wave {info['wave']}, Score: {info['score']}")

        # Render
        if not renderer.render(engine):
            running = False

    renderer.close()


if __name__ == "__main__":
    main()
