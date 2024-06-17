import json
import os


def update_links(notebook_json, path):
    for cell in notebook_json["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [
                x.replace(
                    path, path.replace("_solutions.ipynb", "_skeleton.ipynb")
                ).replace(" - Solution", "")
                for x in cell["source"]
            ]
    return notebook_json


def remove_outputs(notebook_json):
    for cell in notebook_json["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
    return notebook_json


def handle_solutions(notebook_json):
    """
    Take text that looks like
    # BEGIN SOLUTION fitted_dist = ______
    blah blah blah
    # END SOLUTION
    and replace it with
    fitted_dist = ______
    """
    for cell in notebook_json["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = remove_solution_comments(cell["source"])
    return notebook_json


def remove_solution_comments(cell_source):
    in_solution_block = False
    new_source = []
    for line in cell_source:
        if "# BEGIN SOLUTION" in line:
            new_source.append(line.replace("# BEGIN SOLUTION ", ""))
            in_solution_block = True
        elif "# END SOLUTION" in line:
            assert line.strip() == "# END SOLUTION"
            in_solution_block = False
        elif not in_solution_block:
            new_source.append(line)
    return new_source


def create_skeleton(notebook_json, notebook_path):
    notebook_json = remove_outputs(notebook_json)
    notebook_json = handle_solutions(notebook_json)
    notebook_json = update_links(notebook_json, os.path.basename(notebook_path))
    return notebook_json


def run_notebook(notebook_path):
    assert notebook_path.endswith("_solutions.ipynb")
    with open(notebook_path) as f:
        notebook_json = json.load(f)
    notebook_json = create_skeleton(notebook_json, notebook_path)
    with open(notebook_path.replace("_solutions.ipynb", "_skeleton.ipynb"), "w") as f:
        f.write(json.dumps(notebook_json, indent=1) + "\n")


def run_python(python_path):
    with open(python_path) as f:
        python_code = f.read()
    python_code = python_code.split("\n")
    python_code = remove_solution_comments(python_code)
    python_code = "\n".join(python_code)
    with open(python_path.replace("_solutions.py", "_skeleton.py"), "w") as f:
        f.write(python_code)


def run(path):
    if path.endswith(".ipynb"):
        run_notebook(path)
    elif path.endswith(".py"):
        run_python(path)
    else:
        raise ValueError("Path must be a .ipynb or .py file.")


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
