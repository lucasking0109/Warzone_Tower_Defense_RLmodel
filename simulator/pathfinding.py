"""
A* Pathfinding — 8-directional movement on the game grid.
"""
from __future__ import annotations
import heapq
import math
from typing import Optional, List, Dict, Tuple
import numpy as np
from simulator.game_config import GRID_WIDTH, GRID_HEIGHT


# 8 directions: N, NE, E, SE, S, SW, W, NW
DIRECTIONS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
]
SQRT2 = math.sqrt(2)


def heuristic(a, b):
    """Octile distance heuristic for 8-directional movement."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1) * min(dx, dy)


def find_path(passable, start, goal):
    """
    A* pathfinding on a boolean grid.
    passable[x][y] = True if tile is walkable.
    Returns list of (x, y) from start to goal, or None if no path.
    """
    if not passable[start[0]][start[1]] or not passable[goal[0]][goal[1]]:
        return None

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
                continue
            if not passable[nx][ny]:
                continue

            # Prevent corner-cutting through diagonal walls
            if dx != 0 and dy != 0:
                if not passable[cx + dx][cy] or not passable[cx][cy + dy]:
                    continue

            move_cost = SQRT2 if (dx != 0 and dy != 0) else 1.0
            tentative_g = g_score[current] + move_cost
            neighbor = (nx, ny)

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None


def find_paths_from_spawns(passable, spawn_points, goal):
    """Find paths from all spawn points to goal. Returns dict of spawn -> path."""
    paths = {}
    for sp in spawn_points:
        path = find_path(passable, sp, goal)
        if path is not None:
            paths[sp] = path
    return paths


def has_valid_path(passable, spawn_points, goal):
    """Check if at least one spawn point can reach the goal."""
    for sp in spawn_points:
        if find_path(passable, sp, goal) is not None:
            return True
    return False


# ---------------------------------------------------------------
# BFS variants — 10× faster for multi-spawn maps
# ---------------------------------------------------------------

def _reverse_bfs(passable, goal):
    """
    BFS from goal backwards. Returns (visited, parent) arrays.
    Uses 8-directional movement with corner-cutting prevention.
    """
    from collections import deque
    W, H = GRID_WIDTH, GRID_HEIGHT
    visited = np.zeros((W, H), dtype=bool)
    parent_x = np.full((W, H), -1, dtype=np.int16)
    parent_y = np.full((W, H), -1, dtype=np.int16)

    gx, gy = goal
    if not passable[gx][gy]:
        return visited, parent_x, parent_y

    visited[gx][gy] = True
    queue = deque()
    queue.append((gx, gy))

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if visited[nx][ny] or not passable[nx][ny]:
                continue
            # Prevent corner-cutting (same logic as A*)
            if dx != 0 and dy != 0:
                if not passable[cx + dx][cy] or not passable[cx][cy + dy]:
                    continue
            visited[nx][ny] = True
            parent_x[nx][ny] = cx
            parent_y[nx][ny] = cy
            queue.append((nx, ny))

    return visited, parent_x, parent_y


def find_paths_from_spawns_bfs(passable, spawn_points, goal):
    """
    Single reverse BFS from goal, then extract paths for all spawns.
    ~10× faster than calling A* 27 times.
    """
    visited, parent_x, parent_y = _reverse_bfs(passable, goal)
    gx, gy = goal
    paths = {}

    for sp in spawn_points:
        sx, sy = sp
        if not visited[sx][sy]:
            continue
        # Trace path from spawn to goal
        path = [sp]
        cx, cy = sx, sy
        while (cx, cy) != (gx, gy):
            px, py = int(parent_x[cx][cy]), int(parent_y[cx][cy])
            path.append((px, py))
            cx, cy = px, py
        paths[sp] = path

    return paths


def has_all_valid_paths_bfs(passable, spawn_points, goal):
    """Check that ALL spawn points can reach the goal (not just one)."""
    visited, _, _ = _reverse_bfs(passable, goal)
    for sp in spawn_points:
        if not visited[sp[0]][sp[1]]:
            return False
    return True
