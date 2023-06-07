from neurosym.search_graph.search_graph import SearchGraph


def bfs(g: SearchGraph):
    """
    Performs a breadth-first search on the given search graph, yielding each node in the order it
    was visited.

    :param g: Search graph to search over
    """
    visited = set()
    queue = [g.initial_node()]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if g.is_goal_node(node):
            yield node
        for child in g.expand_node(node):
            queue.append(child)
