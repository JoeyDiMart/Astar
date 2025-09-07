#!/usr/bin/env python3
"""
A* Pathfinding Assignment Solution

- Reads a graph from a text file in the specified format
- Runs A* with Euclidean edge costs and Euclidean heuristic
- Prints:
  1) Shortest path (labels)
  2) Total distance
  3) Exploration order (A* expansions)

Usage:
    python a_star_assignment.py <input_file>
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
import sys, math, heapq


#  A class to hold the parameters for location and the name and location cannot change
@dataclass(frozen=True)
class City:
    label: str
    x: float
    y: float


#  class for the graph
@dataclass
class Graph:
    cities: Dict[str, City]
    adj: Dict[str, Set[str]]
    start: str
    goal: str


def parse_graph_file(path: str) -> Graph:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if not lines:
        raise ValueError("Empty file or only blank lines.")

    try:
        N = int(lines[0])  # find number of cities
    except Exception as e:
        raise ValueError(f"First line must be an integer count of cities. Got: {lines[0]!r}") from e

    if len(lines) < 1 + N:
        raise ValueError(f"Expected at least {1 + N} lines, got {len(lines)}.")

    cities: Dict[str, City] = {}   # add the cities to the dictionary
    order: List[str] = []
    for i in range(1, 1 + N):
        parts = [p.strip() for p in lines[i].split(",")]
        if len(parts) != 3:
            raise ValueError(f"City line must have 3 comma-separated values. Got: {lines[i]!r}")
        label, sx, sy = parts
        try:
            x = float(sx)
            y = float(sy)
        except Exception as e:
            raise ValueError(f"City coordinates must be numeric. Got: {sx!r}, {sy!r}") from e
        if label in cities:
            raise ValueError(f"Duplicate city label: {label}")
        if not label.isalpha():
            raise ValueError(f"City label must be alphabetic. Got: {label!r}")
        cities[label] = City(label, x, y)
        order.append(label)

    start = order[0]
    goal = order[-1]

    adj: Dict[str, Set[str]] = {label: set() for label in cities.keys()}

    for i in range(1 + N, len(lines)):  # loop through cities and find add the adjacent cities to a list
        parts = [p.strip() for p in lines[i].split(",")]
        if len(parts) != 2:
            raise ValueError(f"Edge line must have 2 comma-separated labels. Got: {lines[i]!r}")
        a, b = parts
        if a not in cities or b not in cities:
            raise ValueError(f"Edge references unknown city: {a!r}, {b!r}")
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)

    return Graph(cities=cities, adj=adj, start=start, goal=goal)


#  find the the distance between two points
def euclidean(a: City, b: City) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


#  function to perform the algorithm
def a_star(graph: Graph) -> Tuple[List[str], List[str], float]:  # Takes in a graph and returns a list of the shortest path, visited cities, and the path cost
    start, goal = graph.start, graph.goal
    cities, adj = graph.cities, graph.adj

    def h(label: str) -> float:
        return euclidean(cities[label], cities[goal])

    open_heap: List[Tuple[float, float, str]] = []
    g_score: Dict[str, float] = {label: math.inf for label in cities}
    parent: Dict[str, Optional[str]] = {label: None for label in cities}
    closed: Set[str] = set()

    g_score[start] = 0.0
    heapq.heappush(open_heap, (h(start), 0.0, start))

    exploration_order: List[str] = []

    while open_heap:
        f_curr, g_curr, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        exploration_order.append(u)
        if u == goal:
            # reconstruct path
            path: List[str] = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, exploration_order, g_score[goal]

        closed.add(u)
        for v in adj[u]:
            if v in closed:
                continue
            tentative_g = g_score[u] + euclidean(cities[u], cities[v])
            if tentative_g < g_score[v]:
                g_score[v] = tentative_g
                parent[v] = u
                fv = tentative_g + h(v)
                heapq.heappush(open_heap, (fv, tentative_g, v))

    return [], exploration_order, math.inf


# args for running this in the terminal
def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python a_star_assignment.py <input_file>")
        return 2
    path = argv[1]
    graph = parse_graph_file(path)  # create the graph by reading the file
    path_labels, exploration_order, total_cost = a_star(graph)  # run Astar algorithm on the graph

    # print desired output
    print(f"Start: {graph.start}")
    print(f"Goal: {graph.goal}")
    if path_labels:
        print("Shortest path:", " -> ".join(path_labels))
        print(f"Total distance: {total_cost:.6f}")
    else:
        print("No path found.")
    print("Exploration order:", ", ".join(exploration_order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
