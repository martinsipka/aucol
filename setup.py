import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aucol",
    version="0.0.1",
    author="Martin Sipka",
    author_email="martinsipka@gmail.com",
    description="Iteratively refinable representation based CVs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinsipka/aucol",
    project_urls={
        "Bug Tracker": "https://github.com/martinsipka/aucol/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.6",
)
