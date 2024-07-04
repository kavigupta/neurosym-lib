from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression
from neurosym.search_graph.depth_computer import DepthComputer
from neurosym.types.type import ListType


class ProbableDepthComputer(DepthComputer):
    """
    Represents a depth computer that accounts for probable height of partial programs
    """

    def initialize(self):
        return 0

    def depth2go(self, node: SExpression) -> int:
        """
        Calculates a lower bound estimate of the depth2go. For a partial program,
        this is approximated by the minimum number of replacement nodes needed for
        each type of hole. 

        :param node: The node to estimate depth2go for.
        :returns height: The minimum possible height of the node if partial programs
            were filled.
        """
        height = 0
        if isinstance(node, Hole):
            # @TODO: Better heuristic than this.
            height += len(node.twe.typ.input_type)
            # List type's require atleast two nodes to be fully symbolic.
            height += bool(isinstance(node.twe.typ.input_type, ListType))
            return height

        for child in node.children:
            height += 1 + self.depth2go(child)
        return height

    def probable_depth(self, node, current_depth):
        return current_depth + self.depth2go(node)

    def increment(self, node, current_depth):
        return current_depth + 1
