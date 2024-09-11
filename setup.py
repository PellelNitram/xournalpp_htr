import os
import sys

import setuptools

## Modifies config.lua to use the appropriate paths
# Get the path of this file
htr_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the config.lua file
config_file = os.path.join(htr_dir, "plugin", "config.lua")

# Fix direction of slashes, needed on Windows
htr_dir = htr_dir.replace("\\", "/")

# Get the path of the Python executable
python_executable = sys.executable.replace("\\", "/")

# Modify the config.lua file
with open(config_file, "r") as f:
    lines = f.readlines()

# Modify the necessary lines in the config.lua file
modified_lines = []
for line in lines:
    if line.startswith("_M.python_executable ="):
        modified_lines.append('_M.python_executable = "' + python_executable + '"\n')
    elif line.startswith("_M.xournalpp_htr_path ="):
        modified_lines.append(
            '_M.xournalpp_htr_path = "' + htr_dir + '/xournalpp_htr/run_htr.py"\n'
        )
    else:
        modified_lines.append(line)

# Write the modified lines back to the config.lua file
with open(config_file, "w") as f:
    f.writelines(modified_lines)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xournalpp_htr",
    version="0.0.1",
    description="Developing handwritten text recognition for Xournal++.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)
