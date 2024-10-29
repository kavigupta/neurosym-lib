import json
import subprocess
import sys
import tempfile


def execute_notebook(filename, suffix, **kwargs):
    with open(filename) as ff:
        nb_out = json.load(ff)

    source_code = ipynb_to_py(nb_out)
    source_code += [suffix]
    source_code = "\n".join(source_code)
    with tempfile.NamedTemporaryFile("w", delete=True) as f:
        with open(f.name, "w") as f:
            f.write(source_code)

        result = subprocess.check_output([sys.executable, f.name], **kwargs)
    return result.decode("utf-8")


def ipynb_to_py(nb_out):
    source_code = [
        "".join(cell["source"]) + "\n\nplt.close()"
        for cell in nb_out["cells"]
        if cell["cell_type"] == "code"
    ]

    source_code = [
        "import matplotlib.pyplot as plt; plt.show = lambda: None"
    ] + source_code

    source_code = "\n".join(source_code)
    source_code = source_code.split("\n")
    source_code = [
        line
        for line in source_code
        if not line.startswith("%") and not line.startswith("!")
    ]

    return source_code
