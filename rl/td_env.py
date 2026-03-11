"""
Warzone Tower Defense — Gymnasium RL Environment
Pure survival focus: tower build/upgrade/sell + wall placement.
Uses action masking for MaskablePPO (sb3-contrib).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from simulator.game_engine import GameEngine
from simulator.game_map import GameMap
from simulator.game_config import (
    TOWER_TYPES, GRID_WIDTH, GRID_HEIGHT, CREEP_TYPES,
    WALL_BLOCK_COST, GROUND_SPAWN_POINTS, GOAL_POINT,
)

# Fixed orderings for one-hot encoding
TOWER_TYPE_LIST = [
    "bulletTower", "cannonTower", "fireTower",
    "shockTower", "laserTower", "artilleryTower", "PlasmaTower",
]
CREEP_TYPE_LIST = [
    "speeder", "mediumTank", "heavyTank",
    "speederBoss", "mediumTankBoss", "heavyTankBoss",
]

MAX_TOWERS = 200
MAX_WALLS = 100
TICKS_PER_STEP = 40  # 2.0 seconds of game time
MAX_STEPS = 2500     # safety truncation (~wave 250)


class TowerDefenseEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- Pre-compute candidate build positions (static, same every episode) ---
        self._candidate_positions = self._compute_candidate_positions()
        self.N_POS = len(self._candidate_positions)
        self._pos_to_idx = {
            pos: idx for idx, pos in enumerate(self._candidate_positions)
        }
        # Numpy arrays for vectorized buildability check
        self._cand_x = np.array([x for x, y in self._candidate_positions], dtype=np.intp)
        self._cand_y = np.array([y for x, y in self._candidate_positions], dtype=np.intp)

        # --- Wall candidate positions (1×1, static) ---
        self._wall_positions = self._compute_wall_positions()
        self.N_WALL_POS = len(self._wall_positions)
        self._wall_x = np.array([x for x, y in self._wall_positions], dtype=np.intp)
        self._wall_y = np.array([y for x, y in self._wall_positions], dtype=np.intp)

        # --- Tower build costs for fast mask check ---
        self._tower_build_costs = [
            TOWER_TYPES[t]["levels"][1]["cost"] for t in TOWER_TYPE_LIST
        ]

        # --- Upgrade reward lookup (DPS gain per dollar, normalized) ---
        # Average DPS/$ across all towers and levels ≈ 0.2661
        AVG_DPS_PER_DOLLAR = 0.0887  # 0.2661 / 3 → 3x upgrade reward
        self._upgrade_reward = {}  # (tower_type, new_level) -> reward
        for t_key in TOWER_TYPE_LIST:
            t = TOWER_TYPES[t_key]
            prev_dps = 0.0
            for lv in range(1, 6):
                cfg = t["levels"][lv]
                dps = cfg["damage"] / (cfg["reload"] / 1000.0)
                cost = cfg["cost"]
                delta_dps = dps - prev_dps
                self._upgrade_reward[(t_key, lv)] = (delta_dps / cost) / AVG_DPS_PER_DOLLAR
                prev_dps = dps

        # --- Action space layout ---
        self.N_TOWER_TYPES = len(TOWER_TYPE_LIST)
        self.ACTION_NOOP = 0
        self.ACTION_BUILD_START = 1
        self.ACTION_BUILD_END = self.N_TOWER_TYPES * self.N_POS  # inclusive
        self.ACTION_UPGRADE_START = self.ACTION_BUILD_END + 1
        self.ACTION_UPGRADE_END = self.ACTION_UPGRADE_START + MAX_TOWERS - 1
        self.ACTION_WALL_START = self.ACTION_UPGRADE_END + 1
        self.ACTION_WALL_END = self.ACTION_WALL_START + self.N_WALL_POS - 1
        self.ACTION_SPACE_N = self.ACTION_WALL_END + 1

        self.action_space = spaces.Discrete(self.ACTION_SPACE_N)

        # --- Observation space ---
        # 15 global + MAX_TOWERS*17 tower slots + 8 enemy summary + 15 wave info
        self.TOWER_FEATURES = 17  # 15 base + 2 (damage_dealt, steps_alive)
        self.OBS_SIZE = 15 + MAX_TOWERS * self.TOWER_FEATURES + 8 + 15
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32
        )

        # Engine (created on reset)
        self.engine = None
        self._buildable_cache = None
        self._wall_buildable_cache = None

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_candidate_positions():
        m = GameMap()
        positions = []
        for x in range(GRID_WIDTH - 1):
            for y in range(GRID_HEIGHT - 1):
                if m.is_buildable(x, y, 2):
                    positions.append((x, y))
        return positions

    @staticmethod
    def _compute_wall_positions():
        m = GameMap()
        spawn_set = set(GROUND_SPAWN_POINTS)
        positions = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if m.is_buildable(x, y, 1):
                    if (x, y) not in spawn_set and (x, y) != tuple(GOAL_POINT):
                        positions.append((x, y))
        return positions

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.engine = GameEngine()

        # Tower slot tracking
        self._tower_slots = {}   # tower.id -> slot_idx
        self._slot_towers = {}   # slot_idx -> tower object
        self._slot_build_step = {}  # slot_idx -> step when tower was built
        self._free_slots = []
        self._next_free_slot = 0

        # Wall tracking
        self._wall_count = 0

        # Buildable caches (invalidated on build/sell/wall)
        self._buildable_cache = None
        self._wall_buildable_cache = None

        # Reward shaping state
        self._prev_wave = 0
        self._step_count = 0
        self._episode_kills = 0
        self._episode_leaks = 0

        # Path length tracking (for observation features)
        self._baseline_path_length = self._compute_avg_path_length()

        obs = self._get_observation()
        info = {
            "wave": 0,
            "cash": self.engine.cash,
            "score": 0,
            "action_space_n": self.ACTION_SPACE_N,
            "n_candidate_positions": self.N_POS,
            "n_wall_positions": self.N_WALL_POS,
        }
        return obs, info

    def step(self, action):
        assert self.engine is not None, "Call reset() first"
        reward = 0.0
        info = {}

        # === 1. Execute action ===
        reward += self._execute_action(action, info)

        # === 2. Advance simulation ===
        step_killed = 0
        step_leaked = 0
        step_boss_killed = 0
        step_boss_leaked = 0

        for _ in range(TICKS_PER_STEP):
            if self.engine.game_over:
                break
            tick_info = self.engine.tick()

            for enemy in tick_info.get("killed", []):
                step_killed += 1
                if enemy.is_boss:
                    step_boss_killed += 1

            for enemy in tick_info.get("leaked", []):
                step_leaked += 1
                if enemy.is_boss:
                    step_boss_leaked += 1

        # === 3. Compute rewards ===
        reward += step_killed * 1.0
        reward += step_boss_killed * 5.0
        reward += step_leaked * (-5.0)
        reward += step_boss_leaked * (-20.0)

        current_wave = self.engine.wave_controller.wave_number
        if current_wave > self._prev_wave and self._prev_wave > 0:
            if step_leaked == 0:
                reward += 3.0 + current_wave * 0.1
        self._prev_wave = current_wave

        if not self.engine.game_over:
            reward += 0.01
            # Upgrade pressure: penalize having cash while towers need upgrading
            if current_wave >= 20:
                upgradable = 0
                for slot_idx, tower in self._slot_towers.items():
                    if tower.can_upgrade() and self.engine.cash >= tower.upgrade_cost():
                        upgradable += 1
                if upgradable > 0:
                    reward -= 0.005 * upgradable

            # Tower efficiency penalty: damage_dealt / total_invested
            # Low efficiency = tower not earning its cost back
            if self._step_count % 10 == 0:
                for slot_idx, tower in self._slot_towers.items():
                    steps_alive = self._step_count - self._slot_build_step.get(slot_idx, 0)
                    if steps_alive >= 50:
                        efficiency = tower.damage_dealt / max(tower.total_invested, 1)
                        if efficiency < 5:  # < 5 damage per $ invested
                            reward -= 0.01

        if self.engine.game_over:
            reward -= 50.0

        # === 4. Termination / truncation ===
        terminated = self.engine.game_over
        truncated = self._step_count >= MAX_STEPS

        self._step_count += 1
        self._episode_kills += step_killed
        self._episode_leaks += step_leaked

        # === 5. Build observation ===
        obs = self._get_observation()

        info.update({
            "wave": current_wave,
            "cash": self.engine.cash,
            "score": self.engine.score,
            "base_hp": self.engine.base_hp,
            "step_kills": step_killed,
            "step_leaks": step_leaked,
            "episode_kills": self._episode_kills,
            "episode_leaks": self._episode_leaks,
            "num_towers": len(self.engine.towers),
            "wall_count": self._wall_count,
            "step_count": self._step_count,
        })

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        mask = np.zeros(self.ACTION_SPACE_N, dtype=np.int8)

        # NOOP always valid
        mask[self.ACTION_NOOP] = 1

        if self.engine is None or self.engine.game_over:
            return mask

        cash = self.engine.cash
        num_towers = len(self.engine.towers)

        # --- BUILD masks (vectorized) ---
        if num_towers < MAX_TOWERS:
            if self._buildable_cache is None:
                self._recompute_buildable_cache()
            buildable = self._buildable_cache

            for type_idx in range(self.N_TOWER_TYPES):
                cost = self._tower_build_costs[type_idx]
                if cash < cost:
                    continue
                base = self.ACTION_BUILD_START + type_idx * self.N_POS
                mask[base:base + self.N_POS] = buildable

        # --- UPGRADE masks ---
        for slot_idx in range(MAX_TOWERS):
            tower = self._slot_towers.get(slot_idx)
            if tower is not None and tower.can_upgrade():
                if cash >= tower.upgrade_cost():
                    mask[self.ACTION_UPGRADE_START + slot_idx] = 1

        # --- WALL masks (vectorized) ---
        if self._wall_count < MAX_WALLS and cash >= WALL_BLOCK_COST:
            if self._wall_buildable_cache is None:
                self._recompute_wall_buildable_cache()
            mask[self.ACTION_WALL_START:self.ACTION_WALL_START + self.N_WALL_POS] = \
                self._wall_buildable_cache

        return mask

    def _recompute_buildable_cache(self):
        grid = self.engine.map.grid
        cx, cy = self._cand_x, self._cand_y
        self._buildable_cache = (
            (grid[cx, cy] == 0) &
            (grid[cx + 1, cy] == 0) &
            (grid[cx, cy + 1] == 0) &
            (grid[cx + 1, cy + 1] == 0)
        ).astype(np.int8)

    def _recompute_wall_buildable_cache(self):
        grid = self.engine.map.grid
        wx, wy = self._wall_x, self._wall_y
        self._wall_buildable_cache = (grid[wx, wy] == 0).astype(np.int8)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action, info):
        reward = 0.0

        if action == self.ACTION_NOOP:
            info["action_type"] = "noop"

        elif self.ACTION_BUILD_START <= action <= self.ACTION_BUILD_END:
            build_idx = action - self.ACTION_BUILD_START
            type_idx = build_idx // self.N_POS
            pos_idx = build_idx % self.N_POS
            tower_type = TOWER_TYPE_LIST[type_idx]
            x, y = self._candidate_positions[pos_idx]

            success = self.engine.build_tower(tower_type, x, y)
            if success:
                new_tower = self.engine.towers[-1]
                self._allocate_slot(new_tower)
                self._buildable_cache = None
                self._wall_buildable_cache = None
                # No instant build reward — tower earns reward through damage efficiency
                info["action_type"] = "build_ok"
                info["built"] = (tower_type, x, y)
            else:
                reward -= 0.5
                info["action_type"] = "build_fail"

        elif self.ACTION_UPGRADE_START <= action <= self.ACTION_UPGRADE_END:
            slot_idx = action - self.ACTION_UPGRADE_START
            tower = self._slot_towers.get(slot_idx)
            if tower is not None:
                old_level = tower.level
                success = self.engine.upgrade_tower(tower)
                if success:
                    new_level = tower.level
                    reward += self._upgrade_reward.get(
                        (tower.tower_type, new_level), 0.5
                    )
                    info["action_type"] = "upgrade_ok"
                else:
                    reward -= 0.5
                    info["action_type"] = "upgrade_fail"
            else:
                reward -= 0.5
                info["action_type"] = "upgrade_no_tower"

        elif self.ACTION_WALL_START <= action <= self.ACTION_WALL_END:
            pos_idx = action - self.ACTION_WALL_START
            x, y = self._wall_positions[pos_idx]
            prev_path = self._compute_avg_path_length()
            success = self.engine.place_wall(x, y)
            if success:
                self._wall_count += 1
                self._buildable_cache = None
                self._wall_buildable_cache = None
                # Reward/penalize based on actual path change
                new_path = self._compute_avg_path_length()
                delta = new_path - prev_path
                if delta > 0.5:
                    reward += delta * 0.3  # good wall: lengthened path
                else:
                    reward -= 0.3  # useless wall: no path improvement
                info["action_type"] = "wall_ok"
                info["wall_pos"] = (x, y)
            else:
                reward -= 0.5
                info["action_type"] = "wall_fail"

        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_avg_path_length(self):
        """Average path length from all spawns to goal."""
        paths = self.engine._paths
        if not paths:
            return 0.0
        return sum(len(p) for p in paths.values()) / len(paths)

    def _count_path_tiles_in_range(self, tower):
        """Count unique path tiles within tower's shooting range."""
        paths = self.engine._paths
        if not paths:
            return 0
        tower_px, tower_py = tower.px, tower.py
        r_sq = tower.range * tower.range
        tile_size = 24  # TILE_SIZE
        seen = set()
        for path in paths.values():
            for (gx, gy) in path:
                if (gx, gy) in seen:
                    continue
                # Tile center in pixels
                tx = gx * tile_size + tile_size / 2
                ty = gy * tile_size + tile_size / 2
                dx = tx - tower_px
                dy = ty - tower_py
                if dx * dx + dy * dy <= r_sq:
                    seen.add((gx, gy))
        return len(seen)

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _get_observation(self):
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)
        eng = self.engine
        wc = eng.wave_controller
        idx = 0

        # --- Global features (15) ---
        obs[idx] = min(eng.cash / 20000.0, 1.0);               idx += 1
        obs[idx] = eng.base_hp / eng.base_max_hp;               idx += 1
        obs[idx] = min(eng.score / 100000.0, 1.0);              idx += 1
        obs[idx] = min(wc.wave_number / 250.0, 1.0);            idx += 1
        obs[idx] = min(eng.game_time_ms / 5000000.0, 1.0);      idx += 1
        # Wave state one-hot (3)
        state_map = {"waiting_first": 0, "spawning": 1, "between_waves": 2}
        si = state_map.get(wc.state, 0)
        obs[idx + si] = 1.0;                                     idx += 3
        obs[idx] = len(eng.towers) / MAX_TOWERS;                 idx += 1
        obs[idx] = min(len(eng.enemies) / 40.0, 1.0);            idx += 1
        total_inv = sum(t.total_invested for t in eng.towers)
        obs[idx] = min(total_inv / 50000.0, 1.0);                idx += 1

        # Diversity & path features (4) — kept for obs compatibility
        unique_types = len(set(t.tower_type for t in eng.towers)) if eng.towers else 0
        obs[idx] = unique_types / self.N_TOWER_TYPES;             idx += 1
        avg_path = self._compute_avg_path_length()
        obs[idx] = min(avg_path / 100.0, 1.0);                   idx += 1
        obs[idx] = self._wall_count / MAX_WALLS;                  idx += 1
        if self._baseline_path_length > 0:
            obs[idx] = min((avg_path - self._baseline_path_length)
                           / self._baseline_path_length, 1.0)
        idx += 1

        # --- Per-tower slots (MAX_TOWERS × 17) ---
        for slot_idx in range(MAX_TOWERS):
            tower = self._slot_towers.get(slot_idx)
            if tower is not None:
                obs[idx] = 1.0  # exists
                ti = TOWER_TYPE_LIST.index(tower.tower_type)
                obs[idx + 1 + ti] = 1.0
                obs[idx + 8] = tower.level / 5.0
                obs[idx + 9] = tower.grid_x / GRID_WIDTH
                obs[idx + 10] = tower.grid_y / GRID_HEIGHT
                obs[idx + 11] = 1.0 if tower.can_upgrade() else 0.0
                obs[idx + 12] = min(tower.upgrade_cost() / 3200.0, 1.0) if tower.can_upgrade() else 0.0
                obs[idx + 13] = min(tower.sell_value / 5000.0, 1.0)
                obs[idx + 14] = min(tower.total_invested / 10000.0, 1.0)
                # New: damage dealt & steps alive
                obs[idx + 15] = min(tower.damage_dealt / 500000.0, 1.0)
                steps_alive = self._step_count - self._slot_build_step.get(slot_idx, 0)
                obs[idx + 16] = min(steps_alive / 2500.0, 1.0)
            idx += self.TOWER_FEATURES

        # --- Enemy summary (8) ---
        enemies = [e for e in eng.enemies if e.alive and not e.reached_goal]
        n_enemies = len(enemies)

        obs[idx] = min(n_enemies / 40.0, 1.0);  idx += 1

        if n_enemies > 0:
            hp_ratios = [e.hp / e.max_hp for e in enemies]
            obs[idx] = sum(hp_ratios) / n_enemies;                 idx += 1
            obs[idx] = min(hp_ratios);                              idx += 1
            obs[idx] = min(sum(e.hp for e in enemies) / 500000.0, 1.0);  idx += 1
            obs[idx] = min(sum(1 for e in enemies if e.is_boss) / 6.0, 1.0);  idx += 1
            progresses = []
            for e in enemies:
                if len(e.path) > 1:
                    progresses.append(e.path_index / (len(e.path) - 1))
                else:
                    progresses.append(1.0)
            obs[idx] = max(progresses);                            idx += 1
            obs[idx] = sum(e.slow_ratio for e in enemies) / n_enemies;  idx += 1
            obs[idx] = sum(progresses) / n_enemies;                idx += 1
        else:
            idx += 7

        # --- Wave info (15) ---
        if wc.current_wave_config:
            ai_type = wc.current_wave_config.get("ai_type", "")
            if ai_type in CREEP_TYPE_LIST:
                ci = CREEP_TYPE_LIST.index(ai_type)
                obs[idx + ci] = 1.0
        idx += 6

        next_wave = wc.wave_number + 1
        try:
            next_config = wc.get_wave_config(next_wave)
            next_ai = next_config.get("ai_type", "")
            if next_ai in CREEP_TYPE_LIST:
                ni = CREEP_TYPE_LIST.index(next_ai)
                obs[idx + ni] = 1.0
        except (IndexError, KeyError):
            pass
        idx += 6

        if wc.current_wave_config:
            ai_type = wc.current_wave_config.get("ai_type", "")
            est_hp = wc.get_creep_hp(ai_type) if ai_type in CREEP_TYPES else 0
            obs[idx] = min(est_hp / 500000.0, 1.0)
        idx += 1

        if wc.current_wave_config:
            obs[idx] = 1.0 if "Boss" in wc.current_wave_config.get("ai_type", "") else 0.0
        idx += 1

        try:
            next_config = wc.get_wave_config(next_wave)
            obs[idx] = 1.0 if "Boss" in next_config.get("ai_type", "") else 0.0
        except (IndexError, KeyError):
            pass
        idx += 1

        return obs

    # ------------------------------------------------------------------
    # Tower slot management
    # ------------------------------------------------------------------

    def _allocate_slot(self, tower):
        if self._free_slots:
            slot = self._free_slots.pop(0)
        else:
            slot = self._next_free_slot
            self._next_free_slot += 1
        self._tower_slots[tower.id] = slot
        self._slot_towers[slot] = tower
        self._slot_build_step[slot] = self._step_count
        return slot

    def _free_slot(self, slot_idx):
        tower = self._slot_towers.pop(slot_idx, None)
        if tower:
            self._tower_slots.pop(tower.id, None)
        self._free_slots.append(slot_idx)
