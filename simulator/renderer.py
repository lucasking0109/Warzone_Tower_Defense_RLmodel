"""
Pygame Renderer — Visual display of the game state.
"""
import pygame
import math
from simulator.game_config import (
    GRID_WIDTH, GRID_HEIGHT, TILE_SIZE, FPS, COLORS,
    GROUND_SPAWN_POINTS, GOAL_POINT,
)
from simulator.game_map import TileState


HUD_HEIGHT = 80
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + HUD_HEIGHT


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Warzone TD Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)

    def render(self, engine) -> bool:
        """
        Render game state. Returns False if window was closed.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(COLORS["hud_bg"])

        # Draw map
        self._draw_map(engine.map)

        # Draw paths (debug: show path from first spawn)
        if engine._paths:
            first_path = list(engine._paths.values())[0]
            self._draw_path(first_path)

        # Draw towers
        for tower in engine.towers:
            self._draw_tower(tower)

        # Draw enemies
        for enemy in engine.enemies:
            if enemy.alive and not enemy.reached_goal:
                self._draw_enemy(enemy)

        # Draw damage events (flash effects)
        for event in engine.damage_events:
            self._draw_damage_event(event)

        # Draw HUD
        self._draw_hud(engine)

        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def _draw_map(self, game_map):
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile = game_map.grid[x][y]

                if tile == TileState.OBSTACLE:
                    pygame.draw.rect(self.screen, COLORS["obstacle"], rect)
                    pygame.draw.rect(self.screen, COLORS["obstacle_border"], rect, 1)
                elif tile == TileState.WALL:
                    pygame.draw.rect(self.screen, COLORS["wall"], rect)
                elif tile == TileState.TOWER:
                    pass  # drawn separately
                else:
                    pygame.draw.rect(self.screen, COLORS["open"], rect)
                    pygame.draw.rect(self.screen, COLORS["grid_line"], rect, 1)

        # Draw spawn points
        for sx, sy in GROUND_SPAWN_POINTS:
            rect = pygame.Rect(sx * TILE_SIZE, sy * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.screen, COLORS["spawn"], rect)

        # Draw goal
        gx, gy = GOAL_POINT
        rect = pygame.Rect(gx * TILE_SIZE, gy * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.screen, COLORS["goal"], rect)

    def _draw_path(self, path):
        if len(path) < 2:
            return
        points = [
            (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2)
            for x, y in path
        ]
        pygame.draw.lines(self.screen, COLORS["path"], False, points, 1)

    # Tower type abbreviations for on-screen display
    TOWER_ABBREV = {
        "bulletTower": "MG",
        "cannonTower": "CN",
        "fireTower": "FR",
        "shockTower": "SK",
        "laserTower": "LS",
        "artilleryTower": "AT",
        "PlasmaTower": "PL",
    }

    def _draw_tower(self, tower):
        size = tower.block_size * TILE_SIZE
        rect = pygame.Rect(tower.grid_x * TILE_SIZE, tower.grid_y * TILE_SIZE, size, size)

        # Color based on type
        color_map = {
            "bulletTower": COLORS["tower_machine_gun"],
            "cannonTower": COLORS["tower_cannon"],
            "fireTower": COLORS["tower_flame"],
            "shockTower": COLORS["tower_pulse"],
            "laserTower": COLORS["tower_laser"],
            "artilleryTower": COLORS["tower_heavy_cannon"],
            "PlasmaTower": COLORS["tower_plasma"],
        }
        color = color_map.get(tower.tower_type, (100, 100, 100))
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw range circle (faint)
        pygame.draw.circle(
            self.screen, (*color[:3],),
            (int(tower.px), int(tower.py)),
            int(tower.range), 1
        )

        tx = tower.grid_x * TILE_SIZE
        ty = tower.grid_y * TILE_SIZE

        # Level indicator (top-left)
        text = self.font.render(str(tower.level), True, (255, 255, 255))
        self.screen.blit(text, (tx + 2, ty + 1))

        # Type abbreviation (bottom-center)
        abbrev = self.TOWER_ABBREV.get(tower.tower_type, "??")
        abbrev_surf = self.font.render(abbrev, True, (255, 255, 200))
        ax = tx + (size - abbrev_surf.get_width()) // 2
        ay = ty + size - abbrev_surf.get_height() - 2
        self.screen.blit(abbrev_surf, (ax, ay))

    def _draw_enemy(self, enemy):
        # Size based on type
        if enemy.is_boss:
            radius = 6
        elif "heavy" in enemy.creep_type.lower():
            radius = 5
        elif "medium" in enemy.creep_type.lower():
            radius = 4
        else:
            radius = 3

        # Color
        if enemy.is_boss:
            color = COLORS["enemy_boss"]
        elif "heavy" in enemy.creep_type.lower():
            color = COLORS["enemy_heavy"]
        elif "medium" in enemy.creep_type.lower():
            color = COLORS["enemy_medium"]
        else:
            color = COLORS["enemy_speeder"]

        # Tint blue if slowed
        if enemy.slow_timer > 0:
            color = (color[0] // 2, color[1] // 2, min(255, color[2] + 100))

        px, py = int(enemy.px), int(enemy.py)
        pygame.draw.circle(self.screen, color, (px, py), radius)

        # HP bar
        bar_w = radius * 3
        bar_h = 2
        bar_x = px - bar_w // 2
        bar_y = py - radius - 4
        hp_ratio = enemy.hp / enemy.max_hp

        pygame.draw.rect(self.screen, COLORS["hp_bar_bg"],
                         (bar_x, bar_y, bar_w, bar_h))
        fill_color = COLORS["hp_bar_fill"] if hp_ratio > 0.3 else COLORS["hp_bar_low"]
        pygame.draw.rect(self.screen, fill_color,
                         (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _draw_damage_event(self, event):
        enemy = event["enemy"]
        if event["type"] == "aoe":
            tower = event["tower"]
            pygame.draw.circle(
                self.screen, (255, 200, 50),
                (int(enemy.px), int(enemy.py)),
                int(tower.blast_radius), 1
            )
        elif event["type"] == "beam":
            tower = event["tower"]
            pygame.draw.line(
                self.screen, (255, 50, 50),
                (int(tower.px), int(tower.py)),
                (int(enemy.px), int(enemy.py)), 2
            )

    def _draw_hud(self, engine):
        hud_y = GRID_HEIGHT * TILE_SIZE
        pygame.draw.rect(self.screen, COLORS["hud_bg"],
                         (0, hud_y, SCREEN_WIDTH, HUD_HEIGHT))

        wave_state = engine.wave_controller.state
        wave_num = engine.wave_controller.wave_number

        texts = [
            f"Wave: {wave_num}  ({wave_state})",
            f"Cash: ${engine.cash}",
            f"Score: {engine.score}",
            f"Base HP: {engine.base_hp:.1f}/{engine.base_max_hp}",
            f"Enemies: {len(engine.enemies)}",
            f"Time: {engine.game_time_ms // 1000}s",
        ]

        x = 10
        for text in texts:
            surf = self.font.render(text, True, COLORS["text"])
            self.screen.blit(surf, (x, hud_y + 5))
            x += surf.get_width() + 20

        if engine.game_over:
            go_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            self.screen.blit(go_text, (SCREEN_WIDTH // 2 - 60, hud_y + 30))

    def close(self):
        pygame.quit()
