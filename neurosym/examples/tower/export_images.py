# Taken from dreamcoder's tower domain implementation
# https://github.com/CatherineWong/laps_dreamcoder/tree/e5e02131d77f8682ea5ea0c224be0631601b5c09/domains/tower
# Edits have been made, but only extremely minor ones

import os

import imageio
import numpy as np

from neurosym.examples.tower.dsl import tower_dsl
from neurosym.examples.tower.examples import dreamcoder_tower_tasks
from neurosym.examples.tower.renderer import render_plan
from neurosym.examples.tower.state import TowerState


def _make_nice_array(l, columns=None):
    n = columns or int(len(l) ** 0.5)
    a = []
    while l:
        a.append(l[:n])
        l = l[n:]
    return a


def _montage_matrix(matrix):
    import numpy as np

    arrays = matrix
    m = max(len(t) for t in arrays)

    size = arrays[0][0].shape
    tp = arrays[0][0].dtype

    arrays = [
        np.concatenate(ts + [np.zeros(size, dtype=tp)] * (m - len(ts)), axis=1)
        for ts in arrays
    ]
    arrays = np.concatenate(arrays, axis=0)
    return arrays


def _montage(arrays, columns=None):
    return _montage_matrix(_make_nice_array(arrays, columns=columns))


def _write_image(f, a):
    imageio.imwrite(f, (a * 255).astype(np.uint8))


def _execute_tower(program):
    state, plan = tower_dsl().compute(tower_dsl().initialize(program))(TowerState())
    return state.hand, plan


def export_tower_image(program, f, draw_hand=False):

    hand, plan = _execute_tower(program)

    a = render_plan(
        plan,
        pretty=True,
        Lego=True,
        drawHand=hand if draw_hand else None,
    )

    _write_image(f, a)


def export_tower_images(programs, f, draw_hand=False):
    a = _montage(
        [
            render_plan(
                _execute_tower(p)[1],
                pretty=True,
                Lego=True,
                drawHand=_execute_tower(p)[0] if draw_hand else None,
                resolution=256,
            )
            for p in programs
        ]
    )
    _write_image(f, a)


def export_all_tasks(folder):
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    ts = dreamcoder_tower_tasks()
    export_tower_images(
        [t.program for t in ts], os.path.join(folder, "every_tower.png")
    )

    for j, t in enumerate(ts):
        export_tower_image(
            t.program, os.path.join(folder, f"tower_{j}.png"), draw_hand=False
        )

    exampleTowers = [103, 104, 105, 93, 73, 50, 67, 35, 43, 106]
    export_tower_images(
        [ts[n].program for n in exampleTowers],
        os.path.join(folder, "tower_montage.png"),
    )
