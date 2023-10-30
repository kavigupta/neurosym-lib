from dataclasses import dataclass
from typing import Dict, Generic, List, Set, Tuple, TypeVar

K, V = TypeVar("K"), TypeVar("V")


@dataclass
class TreeTrie(Generic[K, V]):
    """
    Like a Trie, but allows lookup of trees
    """

    leaves: Dict[K, List[V]]
    wild: List[V]
    # trie_children[k] is a tree trie for each child of k
    trie_children: Dict[K, Tuple["TreeTrie[K, V]"]]

    @classmethod
    def empty(cls):
        return cls({}, [], {})

    def insert(self, key, value, *, is_wildcard_predicate):
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

    def query(self, key) -> Set[V]:
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
