import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="edge-probing-nonliteral",
    version="1.0.0",
    author="Alex Liebscher",
    author_email="aliebsch@ucsd.edu",
    description="Edge Probing on Contextual Embeddings on Nonliteral Semantics Tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liebscher/edge-probing-nonliteral/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)