import heapq
from typing import List, Tuple, TypeVar, Dict

from tilsdk.localization import *


T = TypeVar('T')

class NoPathFoundException(Exception):
    '''Raise this exception when the pathfinding algorithm cannot reach the endpoint from the startpoint.
    '''
    pass

class InvalidStartException(NoPathFoundException):
    '''This is a more specific NoPathFoundException arising from an invalid start point, i.e. the 
    start coordinates are either off-map or in obstacles.
    '''
    pass


class PriorityQueue:
    # A priority queue where the smaller the value, the higher the priority.
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def is_empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


class Planner:
    def __init__(self, map_:SignedDistanceGrid=None, sdf_weight:float=0.0):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        '''
        self.map = map_
        self.sdf_weight = sdf_weight

    def update_map(self, map:SignedDistanceGrid):
        '''Update planner with new map.'''
        self.map = map

    def heuristic(self, a:GridLocation, b:GridLocation) -> float:
        '''heuristic function for A* pathfinding.
        
        Parameters
        ----------
        a: GridLocation
            Starting location.
        b: GridLocation
            Goal location.
        '''
        return euclidean_distance(a, b)
    
    def plan(self, start:RealLocation, goal:RealLocation) -> List[RealLocation]:
        '''Plan in real coordinates.
        
        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: RealLocation
            Starting location.
        goal: RealLocation
            Goal location.
        
        Returns
        -------
        path
            List of RealLocation from start to goal.
        '''
        print(f"[PLANNER] START:{start.x:.2f},{start.y:.2f}; GOAL: {goal.x:.2f},{goal.y:.2f}")
        path = self.plan_grid(self.map.real_to_grid(start), self.map.real_to_grid(goal))
        return [self.map.grid_to_real(wp) for wp in path]
    
    
    def is_valid_position(self, start: GridLocation):
        '''Checks if discrete position of robot is valid with respect to map layout. 
        Returns False when grid location is out of map, or collides with obstacles.
        '''
        if self.map.in_bounds(start) and self.map.passable(start):
            return True
        else:
            return False


    def plan_grid(self, start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Plan in grid coordinates.

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
            When 
        
        '''

        if not self.map:
            raise RuntimeError('Planner map is not initialized.')
        
        if not self.is_valid_position(start):
            raise InvalidStartException

        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from: Dict[GridLocation, GridLocation] = {}
        cost_so_far: Dict[GridLocation, float] = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.is_empty():
            current: GridLocation = frontier.get()

            if current == goal:
                break

            for next, step_cost, sdf in self.map.neighbours(current):
                new_cost = cost_so_far[current] + step_cost + self.sdf_weight*(1/(sdf))
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current
            
            #print(f'came from: {came_from}')
            #print(f'goal: {goal}')
        
        if goal not in came_from:
            raise NoPathFoundException

        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self,
                         came_from:Dict[GridLocation, GridLocation],
                         start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Traces traversed locations to reconstruct path.
        
        Parameters
        ----------
        came_from: dict
            Dictionary mapping location to location the planner came from.
        start: GridLocation
            Start location for path.
        goal: GridLocation
            Goal location for path.

        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''
        
        current: GridLocation = goal
        path: List[GridLocation] = []
        
        while current != start:
            path.append(current)
            current = came_from[current]
            
        # path.append(start)
        path.reverse()
        return path