"""Grid based path planner."""

import heapq
from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np
from tilsdk.localization import (
    GridLocation,
    RealLocation,
    SignedDistanceGrid,
    euclidean_distance,
)

from til_23_finals.types import LocOrPose
from til_23_finals.utils import ManhattanSDGrid


class NoPathFoundException(Exception):
    """Raise this exception when the pathfinding algorithm cannot reach the endpoint from the startpoint."""

    pass


class InvalidStartException(NoPathFoundException):
    """Specific NoPathFoundException arising from an invalid start point.

    For example, if the start coordinates are either off-map or in obstacles.
    """

    pass


T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """Priority queue implementation."""

    # A priority queue where the smaller the value, the higher the priority.
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def is_empty(self) -> bool:
        """Return True if queue is empty."""
        return not self.elements

    def put(self, item: T, priority: float):
        """Push item into queue with priority."""
        heapq.heappush(self.elements, (priority, item))

    def pop(self) -> T:
        """Pop item with highest priority from queue."""
        return heapq.heappop(self.elements)[1]


class GridPlanner:
    """Grid based path planner."""

    def __init__(self, arena_map: SignedDistanceGrid):
        """Initialize planner.

        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        """
        self.map = ManhattanSDGrid.from_old_class(arena_map)

    def heuristic(self, a: GridLocation, b: GridLocation) -> float:
        """Heuristic function for A* pathfinding.

        Parameters
        ----------
        a: GridLocation
            Starting location.
        b: GridLocation
            Goal location.
        """
        # return euclidean_distance(a, b)
        return abs(a.x - b.x) + abs(a.y - b.y)

    def plan(self, start: LocOrPose, goal: LocOrPose) -> List[RealLocation]:
        """Plan in real coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: LocOrPose
            Starting location.
        goal: LocOrPose
            Goal location.

        Returns
        -------
        path
            List of RealLocation from start to goal.
        """
        start_loc = self.map.real_to_grid(RealLocation(start.x, start.y))
        goal_loc = self.map.real_to_grid(RealLocation(goal.x, goal.y))
        path = self.plan_grid(start_loc, goal_loc)
        return [self.map.grid_to_real(wp) for wp in path]

    def is_valid_position(self, start: GridLocation):
        """Check if discrete position of robot is valid with respect to map layout.

        Returns False when grid location is out of map, or collides with obstacles.
        """
        return self.map.in_bounds(start) and self.map.passable(start)

    def plan_grid(
        self,
        start: GridLocation,
        goal: GridLocation,
        w_sdf: float = 500.0,
        w_dist: float = 1.0,
    ) -> List[GridLocation]:
        """Plan in grid coordinates.

        Parameters
        ----------
        start: GridLocation
            Starting location.
        goal: GridLocation
            Goal location.

        Returns
        -------
        path
            List of waypoints (GridLocation) from start to goal.

        Raises
        ------
        NoPathFoundException
            When there is no path from `start` to `goal` and no InvalidStartException.
        InvalidStartException
            When `start` is not a valid position.
        """
        assert self.map is not None, "Planner map is not initialized."
        if not self.is_valid_position(start):
            raise InvalidStartException

        queue: PriorityQueue[GridLocation] = PriorityQueue()
        walks: Dict[GridLocation, GridLocation] = {}
        costs: Dict[GridLocation, float] = {}

        queue.put(start, 0)
        walks[start] = None
        costs[start] = 0

        while not queue.is_empty():
            cur = queue.pop()
            if cur == goal:
                break

            prev = walks[cur]
            if prev is None:
                delta_prev = (0, 0)
            else:
                delta_prev = (1 if cur.x - prev.x else 0, 1 if cur.y - prev.y else 0)

            for next, dist, sdf in self.map.neighbours(cur):
                cost = w_dist * dist + w_sdf / max(sdf, 1e-6)

                delta_next = (1 if next.x - cur.x else 0, 1 if next.y - cur.y else 0)
                if delta_next != delta_prev:
                    cost *= 2  # Penalty for changing direction of movement, larger than sqrt(2)

                new_cost = costs[cur] + cost
                if next not in costs or new_cost < costs[next]:
                    priority = new_cost + self.heuristic(next, goal)
                    queue.put(next, priority)
                    walks[next] = cur
                    costs[next] = new_cost

        if goal not in walks:
            raise NoPathFoundException
        return self.reconstruct_path(walks, start, goal)

    def reconstruct_path(
        self,
        walks: Dict[GridLocation, GridLocation],
        start: GridLocation,
        goal: GridLocation,
    ) -> List[GridLocation]:
        """Traces traversed locations to reconstruct path.

        Parameters
        ----------
        walks: dict
            Dictionary mapping location to location the planner came from.
        start: GridLocation
            Start location for path.
        goal: GridLocation
            Goal location for path.

        Returns
        -------
        path
            List of GridLocation from start to goal.
        """
        ang_thres = 30  # Threshold to be considered path corner.
        dist_thres = 0.2  # Threshold to filter out waypoints that are too close.

        def _dist(a, b):
            a = self.map.grid_to_real(a)
            b = self.map.grid_to_real(b)
            return euclidean_distance(a, b)

        cur = goal
        path = [goal]
        dir = None
        while cur != start:
            # path.append(cur)
            prev = cur
            cur = walks[cur]
            ang = np.degrees(np.arctan2(cur.y - prev.y, cur.x - prev.x))
            if (
                dir is not None
                and abs(ang - dir) > ang_thres
                and _dist(path[-1], cur) > dist_thres
            ):
                path.append(cur)
            dir = ang
        if _dist(path[-1], start) < dist_thres:
            path.pop()
        path.reverse()
        return path
