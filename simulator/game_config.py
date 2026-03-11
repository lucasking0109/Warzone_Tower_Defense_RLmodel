"""
Warzone Tower Defense Extended — Game Configuration
All values extracted from decompiled SWF (config_mj.xml + level_4_ground_only_config.xml)
Mode: Quick Cash (1.5x cash rewards)
"""

# =============================================================================
# Map Settings (Level 4 Ground Only: "Enclave")
# =============================================================================
GRID_WIDTH = 51   # columns (x: 0-50)
GRID_HEIGHT = 31  # rows    (y: 0-30)
TILE_SIZE = 24    # pixels per tile

# Spawn points for ground units — LEFT side (x=0, y=2..28)
GROUND_SPAWN_POINTS = [
    (0, y) for y in range(2, 29)  # 27 spawn points
]

# Goal point (base location) — upper RIGHT side
GOAL_POINT = (45, 5)

# =============================================================================
# Economy — Quick Cash Mode (initCashPerKill × 1.5, cashWaveIncrease × 1.5)
# =============================================================================
INITIAL_CASH = 1000           # same in quick cash mode (×1.0)
INIT_CASH_PER_KILL = 45       # base 30 × 1.5 (quick cash)
CASH_WAVE_INCREASE = 1.5      # base 1 × 1.5 (quick cash)
CASH_VALUE_MULTIPLIER = 0.5   # sell tower refund ratio

# Kill reward formula: (INIT_CASH_PER_KILL + wave_number * CASH_WAVE_INCREASE) * bonus_multiplier

# =============================================================================
# Base (Life System)
# =============================================================================
BASE_INITIAL_HIT_POINTS = 10
BASE_HEAL_RATE = BASE_INITIAL_HIT_POINTS / 2000  # per tick (very slow)

# =============================================================================
# Creep (Enemy) Stats
# =============================================================================
CREEP_INITIAL_HIT_POINTS = 100
INIT_HP_WAVE_INCREASE = 15
HP_WAVE_INCREASE_MULTIPLIER = 1.045

# Spawning
INIT_SPAWN_COUNT = 2
MAX_SPAWN_COUNT = 6
SPAWN_WAVE_MULTIPLIER = 1.025
CREEP_INSERT_INTERVAL = 1000  # ms between spawns within a wave

# Wave timing
TIME_UNTIL_FIRST_WAVE = 10000  # ms
WAVE_WAIT_TIME = 5000          # ms between waves

# Score
INIT_KILL_POINTS = 10
KILL_WAVE_INCREASE = 1

# Creep type definitions
# hpMultiplier applied to current base HP; creepSpeed in tiles/second
CREEP_TYPES = {
    "speeder": {
        "type": "ground",
        "speed": 2.0,
        "hp_multiplier": 0.5,
        "spawn_ratio": 1,
        "damage_multiplier": 1,
        "bonus_multiplier": 1,
    },
    "mediumTank": {
        "type": "ground",
        "speed": 1.5,
        "hp_multiplier": 1.0,
        "spawn_ratio": 1,
        "damage_multiplier": 1,
        "bonus_multiplier": 1,
    },
    "heavyTank": {
        "type": "ground",
        "speed": 1.0,
        "hp_multiplier": 1.5,
        "spawn_ratio": 1,
        "damage_multiplier": 1,
        "bonus_multiplier": 1,
    },
    "speederBoss": {
        "type": "ground",
        "speed": 2.0,
        "hp_multiplier": 2.5,
        "spawn_ratio": 0,  # only 1 spawns
        "damage_multiplier": 5,
        "bonus_multiplier": 10,
    },
    "mediumTankBoss": {
        "type": "ground",
        "speed": 1.5,
        "hp_multiplier": 5.0,
        "spawn_ratio": 0,
        "damage_multiplier": 5,
        "bonus_multiplier": 10,
    },
    "heavyTankBoss": {
        "type": "ground",
        "speed": 1.0,
        "hp_multiplier": 7.5,
        "spawn_ratio": 0,
        "damage_multiplier": 5,
        "bonus_multiplier": 10,
    },
}

# =============================================================================
# Wave Configuration (Enclave — Level 4 Ground Only)
# =============================================================================
# First 10 waves (intro), then waveList cycles (37 waves)
INTRO_WAVES = [
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    {"ai_type": "heavyTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "speeder"},
    {"ai_type": "mediumTank"},
    {"ai_type": "mediumTank"},
    {"ai_type": "heavyTank"},
]

WAVE_LIST = [
    {"ai_type": "speeder"},
    {"ai_type": "speeder"},
    {"ai_type": "mediumTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    # Boss wave 1
    {"ai_type": "heavyTankBoss"},
    {"ai_type": "speeder", "time_override": 15000},
    # Post-boss
    {"ai_type": "heavyTank"},
    {"ai_type": "mediumTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "speeder"},
    {"ai_type": "heavyTank"},
    {"ai_type": "speeder"},
    {"ai_type": "heavyTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    # Boss wave 2
    {"ai_type": "speeder", "time_override": 15000},
    {"ai_type": "speederBoss"},
    # Post-boss
    {"ai_type": "heavyTank"},
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    {"ai_type": "mediumTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "mediumTank"},
    {"ai_type": "speeder"},
    {"ai_type": "mediumTank"},
    {"ai_type": "heavyTank"},
    {"ai_type": "speeder"},
    # Boss wave 3
    {"ai_type": "mediumTank", "time_override": 15000},
    {"ai_type": "mediumTankBoss"},
    # Post-boss
    {"ai_type": "heavyTank"},
    {"ai_type": "mediumTank"},
]

# =============================================================================
# Tower Definitions
# =============================================================================
# Each tower has levels I-V. Cost is upgrade cost (level I cost = build cost).
# damage_type: "direct" (single target), "indirect" (AOE)
# use_falloff: True means damage decreases with distance from center

TOWER_TYPES = {
    "bulletTower": {  # Machine Gun
        "name": "Machine Gun",
        "targeting": "all",
        "block_size": 2,
        "levels": {
            1: {"cost": 250,  "damage": 24,   "damage_type": "direct", "range": 165, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            2: {"cost": 200,  "damage": 48,   "damage_type": "direct", "range": 170, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            3: {"cost": 400,  "damage": 96,   "damage_type": "direct", "range": 175, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            4: {"cost": 800,  "damage": 192,  "damage_type": "direct", "range": 180, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            5: {"cost": 800,  "damage": 288,  "damage_type": "direct", "range": 210, "reload": 250,  "blast_radius": 0,  "use_falloff": False, "critical": 0.1},
        },
    },
    "cannonTower": {  # Cannon
        "name": "Cannon",
        "targeting": "ground",
        "block_size": 2,
        "levels": {
            1: {"cost": 400,   "damage": 180,   "damage_type": "indirect", "range": 180, "reload": 2000, "blast_radius": 50,  "use_falloff": True, "critical": 0},
            2: {"cost": 320,   "damage": 360,   "damage_type": "indirect", "range": 185, "reload": 2000, "blast_radius": 50,  "use_falloff": True, "critical": 0},
            3: {"cost": 640,   "damage": 720,   "damage_type": "indirect", "range": 190, "reload": 2000, "blast_radius": 50,  "use_falloff": True, "critical": 0},
            4: {"cost": 1280,  "damage": 1440,  "damage_type": "indirect", "range": 195, "reload": 2000, "blast_radius": 50,  "use_falloff": True, "critical": 0},
            5: {"cost": 1280,  "damage": 1920,  "damage_type": "indirect", "range": 220, "reload": 1000, "blast_radius": 50,  "use_falloff": True, "critical": 0.25},
        },
    },
    "fireTower": {  # Flame Thrower
        "name": "Flame Thrower",
        "targeting": "ground",
        "block_size": 2,
        "levels": {
            1: {"cost": 600,   "damage": 4,    "damage_type": "indirect", "range": 120, "reload": 150,  "blast_radius": 25, "use_falloff": True, "critical": 0},
            2: {"cost": 480,   "damage": 8,    "damage_type": "indirect", "range": 125, "reload": 150,  "blast_radius": 25, "use_falloff": True, "critical": 0},
            3: {"cost": 960,   "damage": 16,   "damage_type": "indirect", "range": 130, "reload": 150,  "blast_radius": 25, "use_falloff": True, "critical": 0},
            4: {"cost": 1920,  "damage": 32,   "damage_type": "indirect", "range": 135, "reload": 150,  "blast_radius": 25, "use_falloff": True, "critical": 0},
            5: {"cost": 1920,  "damage": 64,   "damage_type": "indirect", "range": 140, "reload": 100,  "blast_radius": 40, "use_falloff": True, "critical": 0.05},
        },
    },
    "shockTower": {  # Pulse Emitter
        "name": "Pulse Emitter",
        "targeting": "all",
        "block_size": 2,
        "levels": {
            1: {"cost": 1000,  "damage": 750,   "damage_type": "indirect", "range": 55,  "reload": 6000, "blast_radius": 85,  "use_falloff": False, "critical": 0},
            2: {"cost": 800,   "damage": 1500,  "damage_type": "indirect", "range": 60,  "reload": 6000, "blast_radius": 90,  "use_falloff": False, "critical": 0},
            3: {"cost": 1600,  "damage": 3000,  "damage_type": "indirect", "range": 65,  "reload": 6000, "blast_radius": 95,  "use_falloff": False, "critical": 0},
            4: {"cost": 3200,  "damage": 6000,  "damage_type": "indirect", "range": 70,  "reload": 6000, "blast_radius": 100, "use_falloff": False, "critical": 0},
            5: {"cost": 3200,  "damage": 9000,  "damage_type": "indirect", "range": 90,  "reload": 6000, "blast_radius": 120, "use_falloff": False, "critical": 0,
                "slow_ratio": 0.75, "slow_duration": 5000},
        },
    },
    "laserTower": {  # Laser Cannon (ground path)
        "name": "Laser Cannon",
        "targeting": "ground",
        "block_size": 2,
        "weapon_type": "beam",
        "levels": {
            1: {"cost": 1000,  "damage": 500,    "damage_type": "direct", "range": 400, "reload": 4000, "blast_radius": 0, "use_falloff": False, "critical": 0, "beam_length": 400},
            2: {"cost": 800,   "damage": 1000,   "damage_type": "direct", "range": 410, "reload": 4000, "blast_radius": 0, "use_falloff": False, "critical": 0, "beam_length": 410},
            3: {"cost": 1600,  "damage": 2000,   "damage_type": "direct", "range": 420, "reload": 4000, "blast_radius": 0, "use_falloff": False, "critical": 0, "beam_length": 420},
            4: {"cost": 3200,  "damage": 4000,   "damage_type": "direct", "range": 430, "reload": 4000, "blast_radius": 0, "use_falloff": False, "critical": 0, "beam_length": 430},
            5: {"cost": 3200,  "damage": 10000,  "damage_type": "direct", "range": 600, "reload": 4000, "blast_radius": 0, "use_falloff": False, "critical": 0, "beam_length": 600},
        },
    },
    "artilleryTower": {  # Heavy Cannon
        "name": "Heavy Cannon",
        "targeting": "ground",
        "block_size": 2,
        "levels": {
            1: {"cost": 1000,  "damage": 1200,   "damage_type": "indirect", "range": 300, "reload": 6000, "blast_radius": 80,  "use_falloff": True, "critical": 0},
            2: {"cost": 800,   "damage": 2400,   "damage_type": "indirect", "range": 310, "reload": 6000, "blast_radius": 80,  "use_falloff": True, "critical": 0},
            3: {"cost": 1600,  "damage": 4800,   "damage_type": "indirect", "range": 320, "reload": 6000, "blast_radius": 80,  "use_falloff": True, "critical": 0},
            4: {"cost": 3200,  "damage": 9600,   "damage_type": "indirect", "range": 330, "reload": 6000, "blast_radius": 80,  "use_falloff": True, "critical": 0},
            5: {"cost": 3200,  "damage": 19200,  "damage_type": "indirect", "range": 360, "reload": 6000, "blast_radius": 120, "use_falloff": True, "critical": 0.5},
        },
    },
    "PlasmaTower": {  # Plasma Cannon
        "name": "Plasma Cannon",
        "targeting": "all",
        "block_size": 2,
        "levels": {
            1: {"cost": 1000,  "damage": 120,    "damage_type": "direct", "range": 180, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            2: {"cost": 800,   "damage": 240,    "damage_type": "direct", "range": 185, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            3: {"cost": 1600,  "damage": 480,    "damage_type": "direct", "range": 190, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            4: {"cost": 3200,  "damage": 960,    "damage_type": "direct", "range": 195, "reload": 500,  "blast_radius": 0,  "use_falloff": False, "critical": 0},
            5: {"cost": 3200,  "damage": 1920,   "damage_type": "indirect", "range": 200, "reload": 500, "blast_radius": 60, "use_falloff": False, "critical": 0.1},
        },
    },
}

# Wall block
WALL_BLOCK_COST = 50
WALL_BLOCK_SIZE = 1  # 1x1 tile

# =============================================================================
# One-time Skills (Bombs)
# =============================================================================
SKILLS = {
    "bomb": {
        "name": "Bomb",
        "cost": 150,
        "damage": 10000,
        "radius": 65,
        "use_falloff": True,  # not explicitly stated, but standardBomb has no useFallOff key → defaults
    },
    "emp": {
        "name": "E.M.P",
        "cost": 500,
        "damage": 0,
        "radius": 100,
        "slow_ratio": 0.5,
        "slow_duration": 10000,
    },
    "nuke": {
        "name": "Nuke",
        "cost": 1000,
        "damage": 100000,
        "radius": 200,
        "use_falloff": True,
    },
}

# =============================================================================
# Simulation Constants
# =============================================================================
TICK_MS = 50       # milliseconds per simulation tick (20 ticks/sec)
FPS = 60           # pygame rendering FPS

# Colors for rendering
COLORS = {
    "background": (210, 195, 165),     # sand color
    "grid_line": (190, 175, 145),
    "obstacle": (180, 155, 110),
    "obstacle_border": (140, 120, 85),
    "wall": (100, 100, 110),
    "open": (210, 195, 165),
    "spawn": (50, 180, 50),
    "goal": (220, 50, 50),
    "path": (180, 220, 180),
    "tower_machine_gun": (80, 80, 80),
    "tower_cannon": (120, 80, 40),
    "tower_flame": (220, 100, 30),
    "tower_pulse": (50, 200, 50),
    "tower_laser": (200, 50, 50),
    "tower_heavy_cannon": (60, 60, 60),
    "tower_plasma": (50, 220, 180),
    "enemy_speeder": (255, 200, 50),
    "enemy_medium": (50, 150, 255),
    "enemy_heavy": (180, 50, 50),
    "enemy_boss": (200, 50, 200),
    "hp_bar_bg": (60, 60, 60),
    "hp_bar_fill": (50, 220, 50),
    "hp_bar_low": (220, 50, 50),
    "text": (255, 255, 255),
    "hud_bg": (30, 30, 30),
}
