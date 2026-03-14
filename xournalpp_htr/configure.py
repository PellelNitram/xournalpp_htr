import os
import sys


def main():
    # Get the path of the xournalpp_htr repo root (two levels up from this file)
    htr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Path to the config.lua file
    config_file = os.path.join(htr_dir, "plugin", "config.lua")

    # Fix direction of slashes, needed on Windows
    htr_dir = htr_dir.replace("\\", "/")

    # Get the path of the Python executable
    python_executable = sys.executable.replace("\\", "/")

    # Modify the config.lua file
    with open(config_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith("_M.python_executable ="):
            modified_lines.append(
                '_M.python_executable = "' + python_executable + '"\n'
            )
        elif line.startswith("_M.xournalpp_htr_path ="):
            modified_lines.append(
                '_M.xournalpp_htr_path = "' + htr_dir + '/xournalpp_htr/run_htr.py"\n'
            )
        else:
            modified_lines.append(line)

    with open(config_file, "w") as f:
        f.writelines(modified_lines)
