"""
Wave Controller — Manages wave progression and enemy spawning.
"""
import math
import random
from simulator.game_config import (
    INTRO_WAVES, WAVE_LIST, CREEP_TYPES,
    CREEP_INITIAL_HIT_POINTS, INIT_HP_WAVE_INCREASE,
    HP_WAVE_INCREASE_MULTIPLIER,
    INIT_SPAWN_COUNT, MAX_SPAWN_COUNT, SPAWN_WAVE_MULTIPLIER,
    CREEP_INSERT_INTERVAL, WAVE_WAIT_TIME, TIME_UNTIL_FIRST_WAVE,
    INIT_CASH_PER_KILL, CASH_WAVE_INCREASE, INIT_KILL_POINTS, KILL_WAVE_INCREASE,
)
from simulator.enemies import Enemy


class WaveController:
    def __init__(self):
        self.wave_number = 0  # current wave (1-indexed when active)
        self.timer = TIME_UNTIL_FIRST_WAVE  # ms until next event
        self.state = "waiting_first"  # waiting_first, spawning, between_waves, done

        # HP scaling state
        self.creep_current_hp = CREEP_INITIAL_HIT_POINTS
        self.current_hp_increase = INIT_HP_WAVE_INCREASE

        # Spawn state
        self.spawn_count = INIT_SPAWN_COUNT
        self.spawned_this_wave = 0
        self.to_spawn_this_wave = 0
        self.spawn_timer = 0

        # Current wave info
        self.current_wave_config = None

    def get_wave_config(self, wave_number: int) -> dict:
        """Get wave config for a given wave number (1-indexed)."""
        idx = wave_number - 1
        if idx < len(INTRO_WAVES):
            return INTRO_WAVES[idx]
        else:
            cycle_idx = (idx - len(INTRO_WAVES)) % len(WAVE_LIST)
            return WAVE_LIST[cycle_idx]

    def get_creep_hp(self, creep_type: str) -> float:
        """Calculate HP for a creep of given type at current wave."""
        hp_mult = CREEP_TYPES[creep_type]["hp_multiplier"]
        return self.creep_current_hp * hp_mult

    def get_spawn_count(self, creep_type: str) -> int:
        """How many creeps to spawn this wave."""
        spawn_ratio = CREEP_TYPES[creep_type]["spawn_ratio"]
        if spawn_ratio == 0:
            return 1  # Boss: single spawn
        count = min(int(self.spawn_count * spawn_ratio), MAX_SPAWN_COUNT)
        return max(1, count)

    def get_kill_reward(self, wave_number: int, bonus_multiplier: float = 1) -> int:
        return int((INIT_CASH_PER_KILL + wave_number * CASH_WAVE_INCREASE) * bonus_multiplier)

    def get_kill_points(self, wave_number: int) -> int:
        return INIT_KILL_POINTS + wave_number * KILL_WAVE_INCREASE

    def get_wait_time(self) -> int:
        """Get wait time before this wave starts."""
        if self.current_wave_config and "time_override" in self.current_wave_config:
            return self.current_wave_config["time_override"]
        return WAVE_WAIT_TIME

    def start_next_wave(self):
        """Advance to next wave and update scaling."""
        self.wave_number += 1

        # Scale HP
        if self.wave_number > 1:
            self.current_hp_increase *= HP_WAVE_INCREASE_MULTIPLIER
            self.creep_current_hp += self.current_hp_increase

        # Scale spawn count
        self.spawn_count *= SPAWN_WAVE_MULTIPLIER

        # Get wave config
        self.current_wave_config = self.get_wave_config(self.wave_number)
        self.to_spawn_this_wave = self.get_spawn_count(
            self.current_wave_config["ai_type"]
        )
        self.spawned_this_wave = 0
        self.spawn_timer = 0
        self.state = "spawning"

    def update(self, dt_ms, paths, active_enemies):
        """
        Update wave controller. Returns list of newly spawned enemies.
        paths: dict of spawn_point -> path (list of tile positions)
        """
        new_enemies = []

        if self.state == "waiting_first":
            self.timer -= dt_ms
            if self.timer <= 0:
                self.start_next_wave()

        elif self.state == "spawning":
            self.spawn_timer -= dt_ms
            while self.spawn_timer <= 0 and self.spawned_this_wave < self.to_spawn_this_wave:
                enemy = self._spawn_one(paths)
                if enemy:
                    new_enemies.append(enemy)
                self.spawned_this_wave += 1
                self.spawn_timer += CREEP_INSERT_INTERVAL

            # Check if wave spawning is complete
            if self.spawned_this_wave >= self.to_spawn_this_wave:
                # Check if all enemies from this wave are dead/gone
                if not any(e.alive and not e.reached_goal for e in active_enemies
                           if hasattr(e, '_wave') and e._wave == self.wave_number):
                    pass  # We don't wait for kills, just wait between waves
                self.state = "between_waves"
                self.timer = self.get_wait_time()

        elif self.state == "between_waves":
            self.timer -= dt_ms
            if self.timer <= 0:
                self.start_next_wave()

        return new_enemies

    def _spawn_one(self, paths):
        """Spawn a single enemy at a random valid spawn point."""
        if not paths:
            return None

        config = self.current_wave_config
        creep_type = config["ai_type"]
        hp = self.get_creep_hp(creep_type)

        # Pick random spawn point that has a valid path
        valid_spawns = list(paths.keys())
        if not valid_spawns:
            return None
        spawn = random.choice(valid_spawns)
        path = paths[spawn]

        enemy = Enemy(creep_type, hp, list(path))
        enemy._wave = self.wave_number
        return enemy
