import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
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
    python_requires=">=3.10",
    install_requires=[
        "frozendict==2.3.8",
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pytorch-lightning",
        "permacache",
        "requests",
        "stitch-core==0.1.25",
        "scikit-learn",
        "s-exp-parser==1.4.0",
        "no_toplevel_code==1.0.0",
        "ast-scope==0.4.4",
        "pathos==0.3.2"
    ],
)
