FROM mambaorg/micromamba
USER root

WORKDIR /lilo
COPY ./lilo /lilo
COPY ./neurosym /lilo/dreamcoder/neurosym
COPY ./tests /lilo/dreamcoder/tests

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update default packages
RUN apt-get -qq update

# Get Ubuntu packages
RUN apt-get install -y -q \
    build-essential \
    curl \
    libzmq3-dev \
    libcairo2-dev \
    pkg-config

# Install vim and OCaml
RUN apt-get install -y -q \
    vim \
    ocaml \
    opam

RUN apt-cache policy ocaml
RUN apt-cache policy opam
RUN opam init --disable-sandboxing --yes
RUN opam switch list-available
# Build the DreamCoder OCaml binaries
WORKDIR /lilo/dreamcoder
RUN make setup-ocaml
RUN make
WORKDIR /lilo

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install lilo environment with micromamba
RUN micromamba create -y -n lilo -f environment.yml

# Make RUN commands use the new environment:
RUN echo "micromamba activate lilo" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# One-off install of pregex
# (Technically requires Python 3.9+ but in practice compatible with Python 3.7)
RUN python3 -m pip install pregex==1.0.0 --ignore-requires-python
RUN python3 -m pip install s-exp-parser plotly pytorch-lightning ast_scope no_toplevel_code openai==0.27.7
