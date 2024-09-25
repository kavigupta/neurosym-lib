# Taken from dreamcoder's tower domain implementation
# https://github.com/CatherineWong/laps_dreamcoder/tree/e5e02131d77f8682ea5ea0c224be0631601b5c09/domains/tower
# Edits have been made


class TowerState:
    def __init__(self, hand=0, orientation=1, history=None):
        # List of (State|Block)
        self.history = history
        self.hand = hand
        self.orientation = orientation

    def __str__(self):
        return f"S(h={self.hand},o={self.orientation})"

    def __repr__(self):
        return str(self)

    def left(self, n):
        return TowerState(
            hand=self.hand - n,
            orientation=self.orientation,
            history=self.history if self.history is None else self.history + [self],
        )

    def right(self, n):
        return TowerState(
            hand=self.hand + n,
            orientation=self.orientation,
            history=self.history if self.history is None else self.history + [self],
        )

    def reverse(self):
        return TowerState(
            hand=self.hand,
            orientation=-1 * self.orientation,
            history=self.history if self.history is None else self.history + [self],
        )

    def move(self, n):
        return TowerState(
            hand=self.hand + n * self.orientation,
            orientation=self.orientation,
            history=self.history if self.history is None else self.history + [self],
        )

    def recordBlock(self, b):
        if self.history is None:
            return self
        return TowerState(
            hand=self.hand, orientation=self.orientation, history=self.history + [b]
        )


class TowerAction(object):
    def __init__(self, x, w, h):
        self.x = x
        self.w = w * 2
        self.h = h * 2

    def __call__(self, s):
        thisAction = [(self.x + s.hand, self.w, self.h)]
        s = s.recordBlock(thisAction[0])
        return s, thisAction
