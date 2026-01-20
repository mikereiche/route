from collections import defaultdict
from typing import List, Tuple, Hashable, Dict, cast

import util
from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch

# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Please read the docstring for `State` in `util.py` for more details and code.
#   > Please read the docstrings for in `mapUtil.py`, especially for the CityMap class

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """
    verbose = 0

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        # cache of a location and the lowest cost.
        # If a subsequent visit does not have a lower cost, there is no reason to explore it
        self.cache: Dict[str, float] = defaultdict(float)
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap
        self.cache[self.startLocation] = 0.0
        self.state = None

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        self.state = State(self.startLocation, 0.0)
        return self.state
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.endTag in self.cityMap.tags[state.location]
        # END_YOUR_CODE

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        """
        Note we want to return a list of *3-tuples* of the form:
            (actionToReachSuccessor: str, successorState: State, cost: float)
        """
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this
        result: List[Tuple[str, State, float]] = []
        curCost = cast(float, state.memory)
        for (label, dist) in self.cityMap.distances[state.location].items():
            if label in self.cache:
                cachedCost = self.cache[label]
                if curCost + dist >= cachedCost:  # new path is not better than cached, skip
                    continue
            # this is a new path or better than cached
            self.cache[label] = curCost + dist
            result.append((label, State(label, curCost + dist), dist))
        return result
        # END_YOUR_CODE

########################################################################################
# Problem 1b: Custom -- Plan a Route through Stanford

def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. 

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "parking_entrance", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = locationFromTag(makeTag("landmark", "oval"), cityMap)
    endTag = makeTag("landmark", "gates")
    # END_YOUR_CODE

    return ShortestPathProblem(startLocation, endTag, cityMap)

########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.

class Mem(Hashable):
    def __hash__(self):
        return sum(x.__hash__() for x in self.path)

    path: list[str]
    cost: float
    waypoints: tuple[str, ...]

    def __init__(self, path: list[str], cost: float, waypoints: tuple[str, ...]):
        self.path = path
        self.cost = cost
        self.waypoints = waypoints

    def __gt__(self, other):
        return self.cost > other.cost or self.path > other.path or self.waypoints > other.waypoints

    def __lt__(self, other):
        return self.cost < other.cost or self.path < other.path or self.waypoints < other.waypoints

class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Hint: naively, your `memory` representation could be a list of all locations visited.
    However, that would be too large of a state space to search over! Think 
    carefully about what `memory` should represent.
    """

    verbose = 0

    def __init__(
            self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        # This is a cache of each visit to a location
        # Each cache entry is a Dict[waypoints_remaining, cost]
        # such that when we visit a location, we can check if it has already been visited with
        # at least the same set of waypoints visited and a lower cost. If that is the case, there is no
        # need to explore it again as it will not give a lower cost or more waypoints satisfied.
        self.locCache: Dict[
            str, Dict[tuple[str, ...], float]] = defaultdict()  # just need cost and waypoints, set path to [location]
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        # And remove tags satisfied from the get-go.
        self.waypointTags = tuple(
            sorted([item for item in waypointTags if item not in self.cityMap.tags[startLocation]]))

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        state = State(self.startLocation,
                      Mem([self.startLocation], 0, self.waypointTags))  # path, cost, waypoints to visit
        locs = defaultdict(type(float))
        locs[self.waypointTags] = 0.0
        self.locCache = defaultdict()
        self.locCache[self.startLocation] = locs
        return state  # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        mem: Mem = cast(Mem, state.memory)
        if len(mem.waypoints) != 0:
            return False
        return self.endTag in self.cityMap.tags[state.location]
        # END_YOUR_CODE

    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
        result: List[Tuple[str, State, float]] = []
        mem = cast(Mem, state.memory)
        for (label, dist) in self.cityMap.distances[state.location].items():
            waypoints = tuple(sorted([item for item in mem.waypoints if item not in self.cityMap.tags[label]]))
            if label in self.locCache:
                locs = self.locCache[label] # This is a list of costs indexed by the remaining waypoints of that cost
                thisBetter = True  # maybe
                for (wypt, locCost) in locs.items():
                    # if cached entry satisfied at least the same waypoints and had a lower cost, this state not better
                    if set(wypt).issubset(set(waypoints)) and locCost <= mem.cost + dist:
                        thisBetter = False
                        break
                if not thisBetter: # then no need to follow
                    continue  # for (label, dist) in self.cityMap.distances[state.location].items():
                else:
                    locs[waypoints] = mem.cost + dist # cache this cost with these remaining waypoints
            else:  # not in locCache, adding
                locs = defaultdict(type(float))
                locs[waypoints] = mem.cost + dist
                self.locCache[label] = locs

            if len(waypoints) != 0 and label == self.endTag:  # no point going to end if still waypoints left
                print("end with waypoints left")
                continue  # for (label, dist) in self.cityMap.distances[state.location].items():
            newPath = mem.path # list(mem.path) # copy if need to modify when using for debugging
            # newPath.append(label)
            result.append((label, State(label, Mem(newPath, mem.cost + dist, waypoints)), dist))

        return result
        # END_YOUR_CODE


########################################################################################
# Problem 2c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem with waypoints using the map of Stanford, 
    specifying your own `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    startLocation = locationFromTag(makeTag("landmark", "gates"), cityMap)
    waypointTags = [makeTag("landmark", "stanford_stadium"), makeTag("amenity", "bicycle_parking")]
    endTag = makeTag("amenity", "food")  # END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return problem.startState()
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return problem.isEnd(state)
            # END_YOUR_CODE

        def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
            newResult: List[Tuple[str, State, float]] = []
            for (succLoc, succState, succCost) in problem.actionSuccessorsAndCosts(state):
                newResult.append(
                    (succLoc, succState, succCost + heuristic.evaluate(succState) - heuristic.evaluate(state)))
            return newResult
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 3b: "straight-line" heuristic for A*

class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """

    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        self.endGeo = self.cityMap.geoLocations[locationFromTag(self.endTag, cityMap)]
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        return computeDistance(self.cityMap.geoLocations[state.location], self.endGeo)
        # END_YOUR_CODE


########################################################################################
# Problem 3c: "no waypoints" heuristic for A*

class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """

    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        self.endTag = endTag
        self.cityMap = cityMap
        # Precompute
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        self.ecache = defaultdict(float)
        self.endLocs = [location for location, tags in cityMap.tags.items() if endTag in tags]

        # END_YOUR_CODE

        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        class ReverseShortestPathProblem(SearchProblem):
            def __init__(self, endLocs: List[str], endTag: str, cityMap: CityMap):
                self.endLocs = endLocs
                self.cityMap = cityMap
                self.cache = defaultdict(float)

            def startState(self) -> State:
                """
                Return special "END" state
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                return State("END", 0.0)
                # END_YOUR_CODE

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False), 
                UCS will exhaustively compute costs to *all* other states.
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                return False
                # END_YOUR_CODE

            def actionSuccessorsAndCosts(
                    self, state: State
            ) -> List[Tuple[str, State, float]]:
                # If current location is the special "END" state, 
                # return all the locations with the desired endTag and cost 0 
                # (i.e, we connect the special location "END" with cost 0 to all locations with endTag)
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
                result: List[Tuple[str, State, float]] = []
                if state.location == "END":
                    for loc in self.endLocs:
                        result.append((loc, State(loc, 0.0), 0.0))
                    return result

                curCost = cast(float, state.memory)
                for (label, dist) in self.cityMap.distances[state.location].items():
                    if label in self.cache:
                        cachedCost = self.cache[label]
                        if curCost + dist >= cachedCost:  # new path is not better than cached, skip
                            continue
                    # this is a new path or better than cached
                    self.cache[label] = curCost + dist
                    result.append((label, State(label, curCost + dist), dist))
                return result
                # END_YOUR_CODE

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.

        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        ucs = util.UniformCostSearch(verbose=0)
        baseProblem = ReverseShortestPathProblem(self.endLocs, self.endTag, self.cityMap)
        ucs.solve(baseProblem)
        # END_YOUR_CODE

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        for loc, cost in ucs.pastCosts.items():
            self.ecache[loc] = cost

        print("done")
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.ecache[state.location]
        # END_YOUR_CODE
