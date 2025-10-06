from dataclasses import dataclass
from typing import Dict, Generic, List, Set, Tuple, TypeVar

K, V = TypeVar("K"), TypeVar("V")
NS = Tuple[str, int]


@dataclass
class TreeTrie(Generic[K, V]):
    """
    Like a Trie, but allows lookup of trees

    :field leaves: A mapping from the summary of a node and the number of children to the values at that node.
    :field wild: A list of values that are at wildcard nodes.
    :field trie_children: A mapping from the summary of a node and the number of children to the TreeTrie
        for each child of the node.
    """

    leaves: Dict[NS, List[V]]
    wild: List[V]
    # trie_children[k] is a tree trie for each child of k
    trie_children: Dict[NS, Tuple["TreeTrie[K, V]"]]

    @classmethod
    def empty(cls):
        """
        Construct an empty TreeTrie.
        """
        return cls({}, [], {})

    def insert(self, key: K, value: V, *, is_wildcard_predicate: bool) -> None:
        """
        Insert a key-value pair into the tree trie. The key is a tree, and the value
        is a value associated with the key.

        :param key: The key to insert.
        :param value: The value to insert.
        """
        children = list(key.children())
        summary = key.node_summary(), len(children)
        if is_wildcard_predicate(key):
            self.wild.append(value)
        if not children:
            self.leaves[summary] = self.leaves.get(summary, []) + [value]
            return
        if summary not in self.trie_children:
            self.trie_children[summary] = tuple(TreeTrie.empty() for _ in children)
        trie_children = self.trie_children[summary]
        for child, trie_child in zip(children, trie_children):
            trie_child.insert(child, value, is_wildcard_predicate=is_wildcard_predicate)

    def query(self, key: K) -> Set[V]:
        """
        Query the tree trie for values associated with a key.

        :param key: The key to query.
        :return: A set of values associated with the key.
        """
        children = list(key.children())
        summary = key.node_summary(), len(children)
        as_leaf = set() if summary not in self.leaves else set(self.leaves[summary])
        as_wild = set(self.wild)
        if summary in self.trie_children:
            as_child = None
            trie_children = self.trie_children[summary]
            assert len(trie_children) == len(children)
            for child, trie_child in zip(children, trie_children):
                query_finding = trie_child.query(child)
                if as_child is None:
                    as_child = query_finding
                else:
                    as_child = as_child & query_finding
                if as_child is not None and not as_child:
                    break
        else:
            as_child = set()
        return as_leaf | as_wild | as_child
