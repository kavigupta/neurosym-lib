import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neurosym",
    version="0.0.4",
    author="Kavi Gupta",
    author_email="kavig+neurosym@mit.edu",
    description="Neurosymbolic library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kavigupta/neurosym-lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "frozendict==2.3.8",
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pytorch-lightning",
        "permacache",
        "stitch-core",
        "scikit-learn",
        "s-exp-parser @ git+https://github.com/kavigupta/s-exp-parser@bfb01df5df8f61486cdddeeea45f10d962faa7e2",
    ],
)
