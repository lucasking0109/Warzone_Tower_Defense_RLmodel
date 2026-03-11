"""
Enemy (Creep) System — Enemies that follow paths toward the goal.
"""
import math
import random
from simulator.game_config import TILE_SIZE, CREEP_TYPES, TICK_MS


class Enemy:
    _next_id = 0

    def __init__(self, creep_type: str, hit_points: float, path: list[tuple[int, int]]):
        Enemy._next_id += 1
        self.id = Enemy._next_id
        self.creep_type = creep_type
        self.type_data = CREEP_TYPES[creep_type]

        self.max_hp = hit_points
        self.hp = hit_points
        self.speed = self.type_data["speed"]  # tiles per second
        self.damage_multiplier = self.type_data["damage_multiplier"]
        self.bonus_multiplier = self.type_data["bonus_multiplier"]

        # Path following
        self.path = path  # list of (x, y) tile positions
        self.path_index = 0
        # Pixel position (center of tile)
        start_tile = path[0]
        self.px = start_tile[0] * TILE_SIZE + TILE_SIZE / 2
        self.py = start_tile[1] * TILE_SIZE + TILE_SIZE / 2

        self.alive = True
        self.reached_goal = False

        # Slow effect
        self.slow_ratio = 1.0  # 1.0 = normal, lower = slower
        self.slow_timer = 0

    @property
    def is_boss(self) -> bool:
        return "Boss" in self.creep_type

    @property
    def tile_x(self) -> int:
        return int(self.px // TILE_SIZE)

    @property
    def tile_y(self) -> int:
        return int(self.py // TILE_SIZE)

    def apply_damage(self, damage: float, critical_chance: float = 0) -> float:
        """Apply damage. Returns actual damage dealt."""
        if not self.alive:
            return 0
        if critical_chance > 0 and random.random() <= critical_chance:
            damage *= 2
        actual = min(damage, self.hp)
        self.hp -= actual
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
        return actual

    def apply_slow(self, ratio: float, duration_ms: int):
        """Apply slow effect. ratio < 1 means slower."""
        self.slow_ratio = ratio
        self.slow_timer = duration_ms

    def update(self, dt_ms: int):
        """Move enemy along path. dt_ms = time since last update in ms."""
        if not self.alive or self.reached_goal:
            return

        # Update slow
        if self.slow_timer > 0:
            self.slow_timer -= dt_ms
            if self.slow_timer <= 0:
                self.slow_timer = 0
                self.slow_ratio = 1.0

        # Movement speed in pixels per ms
        effective_speed = self.speed * self.slow_ratio
        pixels_per_ms = effective_speed * TILE_SIZE / 1000.0
        distance_to_move = pixels_per_ms * dt_ms

        while distance_to_move > 0 and self.path_index < len(self.path) - 1:
            next_tile = self.path[self.path_index + 1]
            target_px = next_tile[0] * TILE_SIZE + TILE_SIZE / 2
            target_py = next_tile[1] * TILE_SIZE + TILE_SIZE / 2

            dx = target_px - self.px
            dy = target_py - self.py
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= distance_to_move:
                # Reached next waypoint
                self.px = target_px
                self.py = target_py
                self.path_index += 1
                distance_to_move -= dist
            else:
                # Move toward waypoint
                ratio = distance_to_move / dist
                self.px += dx * ratio
                self.py += dy * ratio
                distance_to_move = 0

        # Check if reached goal
        if self.path_index >= len(self.path) - 1:
            self.reached_goal = True

    def distance_to(self, px: float, py: float) -> float:
        """Pixel distance to a point."""
        dx = self.px - px
        dy = self.py - py
        return math.sqrt(dx * dx + dy * dy)
