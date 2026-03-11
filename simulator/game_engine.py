"""
Game Engine — Main simulation loop integrating all systems.
"""
from simulator.game_config import (
    TICK_MS, INITIAL_CASH, BASE_INITIAL_HIT_POINTS, BASE_HEAL_RATE,
    CASH_VALUE_MULTIPLIER, WALL_BLOCK_COST, TOWER_TYPES,
)
from simulator.game_map import GameMap
from simulator.pathfinding import (
    find_paths_from_spawns_bfs, has_all_valid_paths_bfs,
)
from simulator.wave_controller import WaveController
from simulator.towers import Tower


class GameEngine:
    def __init__(self):
        self.map = GameMap()
        self.wave_controller = WaveController()

        self.cash = INITIAL_CASH
        self.score = 0
        self.base_hp = BASE_INITIAL_HIT_POINTS
        self.base_max_hp = BASE_INITIAL_HIT_POINTS

        self.enemies: list = []
        self.towers: list[Tower] = []
        self.damage_events: list[dict] = []  # for rendering

        self.game_over = False
        self.game_time_ms = 0

        # Pre-compute paths
        self._paths = {}
        self._recompute_paths()

    def _recompute_paths(self):
        passable = self.map.get_passable_grid()
        self._paths = find_paths_from_spawns_bfs(
            passable, self.map.spawn_points, self.map.goal
        )

    def _check_path_validity(self) -> bool:
        passable = self.map.get_passable_grid()
        return has_all_valid_paths_bfs(passable, self.map.spawn_points, self.map.goal)

    # === Player Actions ===

    def build_tower(self, tower_type: str, grid_x: int, grid_y: int) -> bool:
        if self.game_over:
            return False
        if tower_type not in TOWER_TYPES:
            return False
        cost = TOWER_TYPES[tower_type]["levels"][1]["cost"]
        if self.cash < cost:
            return False

        tower = Tower(tower_type, grid_x, grid_y)
        if not self.map.is_buildable(grid_x, grid_y, tower.block_size):
            return False

        # Tentatively place and check path validity
        self.map.place_tower(grid_x, grid_y, tower)
        if not self._check_path_validity():
            self.map.remove_tower(grid_x, grid_y)
            return False

        self.cash -= cost
        self.towers.append(tower)
        self._recompute_paths()

        # Update paths for active enemies
        self._update_enemy_paths()
        return True

    def upgrade_tower(self, tower: Tower) -> bool:
        if self.game_over:
            return False
        if not tower.can_upgrade():
            return False
        cost = tower.upgrade_cost()
        if self.cash < cost:
            return False
        self.cash -= cost
        tower.upgrade()
        return True

    def sell_tower(self, tower: Tower) -> bool:
        if self.game_over:
            return False
        if tower not in self.towers:
            return False
        self.cash += tower.sell_value
        self.map.remove_tower(tower.grid_x, tower.grid_y)
        self.towers.remove(tower)
        self._recompute_paths()
        self._update_enemy_paths()
        return True

    def place_wall(self, grid_x: int, grid_y: int) -> bool:
        if self.game_over:
            return False
        if self.cash < WALL_BLOCK_COST:
            return False
        if not self.map.is_buildable(grid_x, grid_y, 1):
            return False

        # Tentatively place wall
        self.map.place_wall(grid_x, grid_y)
        if not self._check_path_validity():
            self.map.remove_wall(grid_x, grid_y)
            return False

        self.cash -= WALL_BLOCK_COST
        self._recompute_paths()
        self._update_enemy_paths()
        return True

    def use_skill(self, skill_name: str, target_px: float, target_py: float) -> bool:
        """Use a one-time bomb/emp/nuke skill."""
        from simulator.game_config import SKILLS
        import math
        if self.game_over:
            return False
        if skill_name not in SKILLS:
            return False
        skill = SKILLS[skill_name]
        if self.cash < skill["cost"]:
            return False

        self.cash -= skill["cost"]

        # Apply to all enemies in radius
        for enemy in self.enemies:
            if not enemy.alive or enemy.reached_goal:
                continue
            dist = math.sqrt((enemy.px - target_px) ** 2 + (enemy.py - target_py) ** 2)
            if dist <= skill["radius"]:
                if skill["damage"] > 0:
                    if skill.get("use_falloff", False):
                        falloff = 1.0 - dist / (skill["radius"] + 8)
                        dmg = skill["damage"] * max(0, falloff)
                    else:
                        dmg = skill["damage"]
                    enemy.apply_damage(dmg)
                if "slow_ratio" in skill:
                    enemy.apply_slow(skill["slow_ratio"], skill["slow_duration"])
        return True

    def _update_enemy_paths(self):
        """Re-route alive enemies with new paths."""
        for enemy in self.enemies:
            if not enemy.alive or enemy.reached_goal:
                continue
            # Find new path from enemy's current tile position
            from simulator.pathfinding import find_path
            passable = self.map.get_passable_grid()
            current_tile = (enemy.tile_x, enemy.tile_y)
            # Make sure current tile is passable for pathfinding
            was_passable = passable[current_tile[0]][current_tile[1]]
            passable[current_tile[0]][current_tile[1]] = True
            new_path = find_path(passable, current_tile, self.map.goal)
            passable[current_tile[0]][current_tile[1]] = was_passable
            if new_path:
                enemy.path = new_path
                enemy.path_index = 0

    # === Simulation ===

    def tick(self) -> dict:
        """
        Advance game by one tick (TICK_MS milliseconds).
        Returns info dict with events from this tick.
        """
        if self.game_over:
            return {"game_over": True}

        dt = TICK_MS
        self.game_time_ms += dt
        self.damage_events.clear()
        killed_this_tick = []
        leaked_this_tick = []

        # 1. Spawn enemies
        new_enemies = self.wave_controller.update(dt, self._paths, self.enemies)
        self.enemies.extend(new_enemies)

        # 2. Move enemies
        for enemy in self.enemies:
            if enemy.alive and not enemy.reached_goal:
                enemy.update(dt)

        # 3. Towers attack
        alive_enemies = [e for e in self.enemies if e.alive and not e.reached_goal]
        for tower in self.towers:
            events = tower.update(dt, alive_enemies)
            self.damage_events.extend(events)

        # 4. Process killed enemies
        for enemy in self.enemies:
            if not enemy.alive and not hasattr(enemy, '_processed'):
                enemy._processed = True
                reward = self.wave_controller.get_kill_reward(
                    self.wave_controller.wave_number, enemy.bonus_multiplier
                )
                self.cash += reward
                self.score += self.wave_controller.get_kill_points(
                    self.wave_controller.wave_number
                )
                killed_this_tick.append(enemy)

        # 5. Process leaked enemies (reached goal)
        for enemy in self.enemies:
            if enemy.reached_goal and not hasattr(enemy, '_leaked'):
                enemy._leaked = True
                self.base_hp -= enemy.damage_multiplier
                leaked_this_tick.append(enemy)

        # 6. Base healing (very slow)
        if self.base_hp > 0:
            self.base_hp = min(self.base_max_hp, self.base_hp + BASE_HEAL_RATE)

        # 7. Check game over
        if self.base_hp <= 0:
            self.base_hp = 0
            self.game_over = True

        # 8. Clean up dead/leaked enemies
        self.enemies = [
            e for e in self.enemies
            if e.alive and not e.reached_goal
        ]

        return {
            "game_over": self.game_over,
            "wave": self.wave_controller.wave_number,
            "spawned": new_enemies,
            "killed": killed_this_tick,
            "leaked": leaked_this_tick,
            "cash": self.cash,
            "score": self.score,
            "base_hp": self.base_hp,
            "enemies_alive": len(self.enemies),
            "time_ms": self.game_time_ms,
        }
