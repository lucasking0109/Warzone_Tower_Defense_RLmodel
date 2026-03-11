"""
Game Map — Grid system with obstacles, walls, towers, spawn/goal points.
Level 4 "Enclave" (ground only). Coordinates: (x, y) where x=column (0=left), y=row (0=top).
"""
import numpy as np
from enum import IntEnum
from simulator.game_config import (
    GRID_WIDTH, GRID_HEIGHT, TILE_SIZE,
    GROUND_SPAWN_POINTS, GOAL_POINT,
)


class TileState(IntEnum):
    OPEN = 0
    OBSTACLE = 1   # natural terrain (rocks, cliffs, border)
    WALL = 2       # player-placed wall block
    TOWER = 3      # player-placed tower (2x2)


# Block points from level_4_ground_only_layout.xml (Enclave)
# These are all (x, y) positions marked as blocked in the original game data.
# Values like C_1, B_C_*, T_C_*, R_C_*, L_C_* are all impassable terrain.
_RAW_BLOCK_POINTS = [
    # x=0
    (0,0),(0,1),(0,29),(0,30),
    # x=1
    (1,0),(1,1),(1,29),(1,30),
    # x=2
    (2,0),(2,1),(2,29),(2,30),
    # x=3
    (3,0),(3,1),(3,29),(3,30),
    # x=4
    (4,0),(4,1),(4,29),(4,30),
    # x=5
    (5,0),(5,1),(5,29),(5,30),
    # x=6
    (6,0),(6,1),(6,29),(6,30),
    # x=7
    (7,0),(7,1),(7,29),(7,30),
    # x=8
    (8,0),(8,1),(8,8),(8,9),(8,29),(8,30),
    # x=9
    (9,0),(9,1),(9,8),(9,9),(9,20),(9,21),(9,29),(9,30),
    # x=10
    (10,0),(10,1),(10,8),(10,9),(10,20),(10,21),(10,29),(10,30),
    # x=11
    (11,0),(11,1),(11,29),(11,30),
    # x=12
    (12,0),(12,1),(12,29),(12,30),
    # x=13
    (13,0),(13,1),(13,29),(13,30),
    # x=14
    (14,0),(14,1),(14,29),(14,30),
    # x=15
    (15,0),(15,1),(15,29),(15,30),
    # x=16
    (16,0),(16,1),(16,29),(16,30),
    # x=17
    (17,0),(17,1),(17,26),(17,27),(17,29),(17,30),
    # x=18
    (18,0),(18,1),(18,14),(18,15),(18,26),(18,27),(18,29),(18,30),
    # x=19
    (19,0),(19,1),(19,13),(19,14),(19,15),(19,16),(19,29),(19,30),
    # x=20
    (20,0),(20,1),(20,13),(20,14),(20,15),(20,16),(20,29),(20,30),
    # x=21
    (21,0),(21,1),(21,13),(21,14),(21,15),(21,16),(21,17),(21,18),(21,29),(21,30),
    # x=22
    (22,0),(22,1),(22,13),(22,14),(22,15),(22,16),(22,17),(22,18),(22,19),(22,29),(22,30),
    # x=23
    (23,0),(23,1),(23,14),(23,15),(23,16),(23,17),(23,18),(23,19),(23,29),(23,30),
    # x=24
    (24,0),(24,1),(24,2),(24,14),(24,15),(24,16),(24,17),(24,18),(24,19),(24,29),(24,30),
    # x=25
    (25,0),(25,1),(25,2),(25,3),(25,4),(25,5),(25,6),(25,16),(25,17),(25,18),(25,19),(25,29),(25,30),
    # x=26
    (26,0),(26,1),(26,2),(26,3),(26,4),(26,5),(26,6),(26,17),(26,18),(26,29),(26,30),
    # x=27
    (27,0),(27,1),(27,2),(27,3),(27,4),(27,5),(27,6),(27,7),(27,8),(27,9),(27,22),(27,23),(27,24),(27,25),(27,26),(27,29),(27,30),
    # x=28
    (28,0),(28,1),(28,2),(28,3),(28,4),(28,5),(28,6),(28,7),(28,8),(28,9),(28,22),(28,23),(28,24),(28,25),(28,26),(28,29),(28,30),
    # x=29
    (29,0),(29,1),(29,2),(29,3),(29,4),(29,5),(29,6),(29,7),(29,8),(29,9),(29,10),(29,20),(29,21),(29,22),(29,23),(29,24),(29,25),(29,26),(29,29),(29,30),
    # x=30
    (30,0),(30,1),(30,5),(30,6),(30,7),(30,8),(30,9),(30,10),(30,20),(30,21),(30,22),(30,23),(30,24),(30,25),(30,26),(30,29),(30,30),
    # x=31
    (31,0),(31,1),(31,5),(31,6),(31,7),(31,8),(31,21),(31,22),(31,23),(31,24),(31,25),(31,26),(31,29),(31,30),
    # x=32
    (32,0),(32,1),(32,7),(32,8),(32,23),(32,24),(32,25),(32,29),(32,30),
    # x=33
    (33,0),(33,1),(33,23),(33,24),(33,29),(33,30),
    # x=34
    (34,0),(34,1),(34,29),(34,30),
    # x=35
    (35,0),(35,1),(35,29),(35,30),
    # x=36
    (36,0),(36,1),(36,15),(36,16),(36,17),(36,29),(36,30),
    # x=37
    (37,0),(37,1),(37,15),(37,16),(37,17),(37,18),(37,29),(37,30),
    # x=38
    (38,0),(38,1),(38,14),(38,15),(38,16),(38,17),(38,18),(38,19),(38,20),(38,29),(38,30),
    # x=39
    (39,0),(39,1),(39,14),(39,15),(39,16),(39,17),(39,18),(39,19),(39,20),(39,29),(39,30),
    # x=40
    (40,0),(40,1),(40,15),(40,16),(40,17),(40,18),(40,19),(40,20),(40,21),(40,29),(40,30),
    # x=41
    (41,0),(41,1),(41,18),(41,19),(41,20),(41,21),(41,29),(41,30),
    # x=42
    (42,0),(42,1),(42,19),(42,20),(42,21),(42,22),(42,23),(42,29),(42,30),
    # x=43
    (43,0),(43,1),(43,19),(43,20),(43,21),(43,22),(43,23),(43,24),(43,29),(43,30),
    # x=44
    (44,0),(44,1),(44,19),(44,20),(44,21),(44,22),(44,23),(44,24),(44,29),(44,30),
    # x=45
    (45,0),(45,1),(45,20),(45,21),(45,22),(45,23),(45,24),(45,25),(45,26),(45,29),(45,30),
    # x=46
    (46,0),(46,1),(46,21),(46,22),(46,23),(46,24),(46,25),(46,26),(46,29),(46,30),
    # x=47
    (47,0),(47,1),(47,23),(47,24),(47,25),(47,26),(47,29),(47,30),
    # x=48
    (48,0),(48,1),(48,2),(48,23),(48,24),(48,25),(48,26),(48,27),(48,28),(48,29),(48,30),
    # x=49
    (49,0),(49,1),(49,2),(49,3),(49,4),(49,5),(49,6),(49,7),(49,8),(49,9),(49,10),
    (49,11),(49,12),(49,13),(49,14),(49,15),(49,16),(49,17),(49,18),(49,19),(49,20),
    (49,21),(49,22),(49,23),(49,24),(49,25),(49,26),(49,27),(49,28),(49,29),(49,30),
    # x=50
    (50,0),(50,1),(50,2),(50,3),(50,4),(50,5),(50,6),(50,7),(50,8),(50,9),(50,10),
    (50,11),(50,12),(50,13),(50,14),(50,15),(50,16),(50,17),(50,18),(50,19),(50,20),
    (50,21),(50,22),(50,23),(50,24),(50,25),(50,26),(50,27),(50,28),(50,29),(50,30),
]


class GameMap:
    def __init__(self):
        # grid[x][y] — TileState
        self.grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.int8)
        self.spawn_points = list(GROUND_SPAWN_POINTS)
        self.goal = GOAL_POINT
        self.towers = {}  # (x, y) -> tower object (top-left corner)
        self._init_obstacles()

    def _init_obstacles(self):
        for (x, y) in _RAW_BLOCK_POINTS:
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                self.grid[x][y] = TileState.OBSTACLE

    def is_passable(self, x: int, y: int) -> bool:
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return False
        return self.grid[x][y] == TileState.OPEN

    def is_buildable(self, x: int, y: int, size: int = 1) -> bool:
        """Check if we can place a structure of given size at (x, y)."""
        for dx in range(size):
            for dy in range(size):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
                    return False
                if self.grid[nx][ny] != TileState.OPEN:
                    return False
                # Can't build on spawn or goal
                if (nx, ny) in self.spawn_points or (nx, ny) == self.goal:
                    return False
        return True

    def place_wall(self, x: int, y: int) -> bool:
        if not self.is_buildable(x, y, 1):
            return False
        self.grid[x][y] = TileState.WALL
        return True

    def remove_wall(self, x: int, y: int) -> bool:
        if self.grid[x][y] == TileState.WALL:
            self.grid[x][y] = TileState.OPEN
            return True
        return False

    def place_tower(self, x: int, y: int, tower) -> bool:
        """Place a 2x2 tower at top-left corner (x, y)."""
        size = tower.block_size
        if not self.is_buildable(x, y, size):
            return False
        for dx in range(size):
            for dy in range(size):
                self.grid[x + dx][y + dy] = TileState.TOWER
        self.towers[(x, y)] = tower
        return True

    def remove_tower(self, x: int, y: int) -> bool:
        if (x, y) not in self.towers:
            return False
        tower = self.towers.pop((x, y))
        size = tower.block_size
        for dx in range(size):
            for dy in range(size):
                self.grid[x + dx][y + dy] = TileState.OPEN
        return True

    def get_passable_grid(self) -> np.ndarray:
        """Return boolean grid where True = passable."""
        return self.grid == TileState.OPEN

    def pixel_to_tile(self, px, py):
        return int(px // TILE_SIZE), int(py // TILE_SIZE)

    def tile_to_pixel(self, x, y):
        """Return center pixel of tile."""
        return x * TILE_SIZE + TILE_SIZE / 2, y * TILE_SIZE + TILE_SIZE / 2
