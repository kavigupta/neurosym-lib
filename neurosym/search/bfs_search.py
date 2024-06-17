from neurosym.search_graph.search_graph import SearchGraph


def bfs(g: SearchGraph, iteration_limit=float("inf")):
    """
    Performs a breadth-first search on the given search graph, yielding each goal node
    in the order it was visited.

    :param g: Search graph to search over
    :param iteration_limit: Maximum number of iterations to perform
    """
    visited = set()
    queue = [g.initial_node()]
    while queue:
        node = queue.pop(0)
        iteration_limit -= 1
        if iteration_limit < 0:
            break
        if node in visited:
            continue
        visited.add(node)
        if g.is_goal_node(node):
            yield node
        for child in g.expand_node(node):
            queue.append(child)
