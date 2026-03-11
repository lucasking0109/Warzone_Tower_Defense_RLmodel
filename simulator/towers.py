"""
Tower System — Tower placement, targeting, and attacking.
"""
import math
from simulator.game_config import TOWER_TYPES, TILE_SIZE


class Tower:
    _next_id = 0

    def __init__(self, tower_type: str, grid_x: int, grid_y: int, level: int = 1):
        Tower._next_id += 1
        self.id = Tower._next_id
        self.tower_type = tower_type
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.level = level

        type_data = TOWER_TYPES[tower_type]
        self.name = type_data["name"]
        self.targeting = type_data["targeting"]
        self.block_size = type_data["block_size"]
        self.is_beam = type_data.get("weapon_type") == "beam"

        # Center pixel position (for 2x2 tower, center is at +1 tile from top-left)
        self.px = (grid_x + self.block_size / 2) * TILE_SIZE
        self.py = (grid_y + self.block_size / 2) * TILE_SIZE

        # Stats from level
        self._update_stats()

        # Combat state
        self.reload_timer = 0  # ms until next shot
        self.target = None     # current enemy target
        self.damage_dealt = 0  # cumulative damage dealt

    def _update_stats(self):
        stats = TOWER_TYPES[self.tower_type]["levels"][self.level]
        self.damage = stats["damage"]
        self.damage_type = stats["damage_type"]
        self.range = stats["range"]
        self.reload_time = stats["reload"]
        self.blast_radius = stats["blast_radius"]
        self.use_falloff = stats["use_falloff"]
        self.critical_chance = stats["critical"]
        self.beam_length = stats.get("beam_length", 0)
        self.slow_ratio = stats.get("slow_ratio", None)
        self.slow_duration = stats.get("slow_duration", 0)

    @property
    def total_invested(self) -> int:
        """Total cost invested including all upgrades."""
        levels = TOWER_TYPES[self.tower_type]["levels"]
        return sum(levels[l]["cost"] for l in range(1, self.level + 1))

    @property
    def sell_value(self) -> int:
        from simulator.game_config import CASH_VALUE_MULTIPLIER
        return int(self.total_invested * CASH_VALUE_MULTIPLIER)

    def can_upgrade(self) -> bool:
        return self.level < 5

    def upgrade_cost(self) -> int:
        if not self.can_upgrade():
            return 0
        return TOWER_TYPES[self.tower_type]["levels"][self.level + 1]["cost"]

    def upgrade(self):
        if self.can_upgrade():
            self.level += 1
            self._update_stats()

    def is_in_range(self, enemy) -> bool:
        dist = self._distance_to_enemy(enemy)
        return dist <= self.range

    def _distance_to_enemy(self, enemy) -> float:
        dx = self.px - enemy.px
        dy = self.py - enemy.py
        return math.sqrt(dx * dx + dy * dy)

    def find_target(self, enemies: list) -> bool:
        """Find closest enemy in range. Returns True if target found."""
        # Check if current target is still valid
        if self.target and self.target.alive and not self.target.reached_goal:
            if self.is_in_range(self.target):
                return True

        # Find new target: closest enemy in range
        best_dist = float("inf")
        best_enemy = None
        for enemy in enemies:
            if not enemy.alive or enemy.reached_goal:
                continue
            if self.targeting == "ground" and enemy.type_data["type"] != "ground":
                continue
            dist = self._distance_to_enemy(enemy)
            if dist <= self.range and dist < best_dist:
                best_dist = dist
                best_enemy = enemy
        self.target = best_enemy
        return self.target is not None

    def update(self, dt_ms: int, enemies: list) -> list[dict]:
        """
        Update tower: find target, fire if ready.
        Returns list of damage events.
        """
        events = []
        self.reload_timer = max(0, self.reload_timer - dt_ms)

        if not self.find_target(enemies):
            return events

        if self.reload_timer > 0:
            return events

        # Fire!
        self.reload_timer = self.reload_time

        if self.is_beam:
            events.extend(self._fire_beam(enemies))
        elif self.blast_radius > 0:
            events.extend(self._fire_aoe(enemies))
        else:
            events.extend(self._fire_direct())

        return events

    def _fire_direct(self) -> list[dict]:
        """Single-target direct damage."""
        if not self.target or not self.target.alive:
            return []
        actual = self.target.apply_damage(self.damage, self.critical_chance)
        self.damage_dealt += actual
        return [{"type": "direct", "tower": self, "enemy": self.target, "damage": actual}]

    def _fire_aoe(self, enemies: list) -> list[dict]:
        """AOE damage centered on target position."""
        if not self.target or not self.target.alive:
            return []

        target_px, target_py = self.target.px, self.target.py
        events = []

        for enemy in enemies:
            if not enemy.alive or enemy.reached_goal:
                continue
            if self.targeting == "ground" and enemy.type_data["type"] != "ground":
                continue
            dist = math.sqrt((enemy.px - target_px) ** 2 + (enemy.py - target_py) ** 2)
            if dist <= self.blast_radius:
                if self.use_falloff:
                    # Damage decreases with distance: damage * (1 - dist / (radius + 8))
                    falloff = 1.0 - dist / (self.blast_radius + 8)
                    actual_damage = self.damage * max(0, falloff)
                else:
                    actual_damage = self.damage

                actual = enemy.apply_damage(actual_damage, self.critical_chance)
                self.damage_dealt += actual
                if self.slow_ratio is not None and self.slow_duration > 0:
                    enemy.apply_slow(self.slow_ratio, self.slow_duration)
                events.append({"type": "aoe", "tower": self, "enemy": enemy, "damage": actual})

        return events

    def _fire_beam(self, enemies: list) -> list[dict]:
        """Beam weapon: hits all enemies along a line from tower to target direction."""
        if not self.target or not self.target.alive:
            return []

        # Direction from tower to target
        dx = self.target.px - self.px
        dy = self.target.py - self.py
        dist = math.sqrt(dx * dx + dy * dy)
        if dist == 0:
            return []
        dx /= dist
        dy /= dist

        beam_len = self.beam_length
        beam_width = 10  # approximate beam width for collision

        events = []
        for enemy in enemies:
            if not enemy.alive or enemy.reached_goal:
                continue
            if self.targeting == "ground" and enemy.type_data["type"] != "ground":
                continue

            # Project enemy position onto beam line
            ex = enemy.px - self.px
            ey = enemy.py - self.py
            proj = ex * dx + ey * dy  # projection along beam
            if proj < 0 or proj > beam_len:
                continue
            # Perpendicular distance
            perp = abs(ex * (-dy) + ey * dx)
            if perp <= beam_width:
                actual = enemy.apply_damage(self.damage, self.critical_chance)
                self.damage_dealt += actual
                events.append({"type": "beam", "tower": self, "enemy": enemy, "damage": actual})

        return events
