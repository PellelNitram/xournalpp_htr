import setuptools

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
