import numpy as np

from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    PreorderMask,
)


class DreamCoderLambdaTransparencyMask(PreorderMask):
    """
    PreorderMask that implements DreamCoder's lambda transparency.

    In DreamCoder, lambdas are structurally transparent: the body of a
    lambda inherits the production distribution of its nearest non-lambda
    ancestor.  The base bigram TreeDistribution (limit=1) uses the
    noParent grammar for all lambda bodies.  This mask adjusts the
    log-probabilities so that lambda bodies effectively use the library
    grammar of their nearest non-lambda ancestor instead.
    """

    def __init__(self, tree_dist):
        """
        :param tree_dist: The TreeDistribution.
        """
        super().__init__(tree_dist)
        self.lam_inds = {
            i for i, (sym, _) in enumerate(tree_dist.symbols)
            if sym.startswith("lam_")
        }
        self.stack = []
        self.adjustments = {}

        if not self.lam_inds:
            return

        # Build noparent log-prob lookup from any lambda parent's distribution
        # (all lambda parents share the same distribution = noParent).
        some_lam = next(iter(self.lam_inds))
        noparent = {
            int(child): ll
            for child, ll in tree_dist.distribution.get(((some_lam, 0),), [])
        }

        # For each non-lambda (parent, arg_pos), compute the per-child
        # adjustment: library_ll - noparent_ll.
        for key, children in tree_dist.distribution.items():
            ((parent_ind, arg_pos),) = key
            parent_ind, arg_pos = int(parent_ind), int(arg_pos)
            if parent_ind in self.lam_inds:
                continue
            adj = {}
            for child, lib_ll in children:
                child = int(child)
                diff = lib_ll - noparent.get(child, 0.0)
                if abs(diff) > 1e-15:
                    adj[child] = diff
            if adj:
                self.adjustments[(parent_ind, arg_pos)] = adj

    def compute_mask(self, position, symbols):
        if not self.stack or self.stack[-1][0] not in self.lam_inds:
            # Not in a lambda body; no adjustment needed.
            return np.zeros(len(symbols))

        effective = self._find_effective_context()
        if effective is None or effective not in self.adjustments:
            return np.zeros(len(symbols))

        adj_dict = self.adjustments[effective]
        result = np.zeros(len(symbols))
        for i, sym in enumerate(symbols):
            sym = int(sym)
            if sym in adj_dict:
                result[i] = adj_dict[sym]
        return result

    def _find_effective_context(self):
        """Scan the stack to find the first non-lambda ancestor.

        Returns (parent_ind, arg_pos) where arg_pos is the position
        of the first lambda in the chain within the parent.
        """
        child_pos = None
        for i in range(len(self.stack) - 1, -1, -1):
            sym, pos = self.stack[i]
            if sym not in self.lam_inds:
                if child_pos is not None:
                    return (sym, child_pos)
                return (sym, pos)
            child_pos = pos
        return None

    def on_entry(self, position, symbol):
        self.stack.append((symbol, position))
        def undo():
            self.stack.pop()
        return undo

    def on_exit(self, position, symbol):
        return lambda: None

    @property
    def can_cache(self):
        return True

    def cache_key(self, parents):
        if self.stack and self.stack[-1][0] in self.lam_inds:
            return self._find_effective_context()
        return None
