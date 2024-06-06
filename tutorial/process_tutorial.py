import json


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


def create_skeleton(notebook_json):
    notebook_json = remove_outputs(notebook_json)
    notebook_json = handle_solutions(notebook_json)
    return notebook_json


def run_notebook(notebook_path):
    assert notebook_path.endswith("_solutions.ipynb")
    with open(notebook_path) as f:
        notebook_json = json.load(f)
    notebook_json = create_skeleton(notebook_json)
    with open(notebook_path.replace("_solutions.ipynb", ".ipynb"), "w") as f:
        json.dump(notebook_json, f)


if __name__ == "__main__":
    import sys

    run_notebook(sys.argv[1])
