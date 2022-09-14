# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

MAINTAINER marcdomenechvila@gmail.com

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"


# INSTALL COMMANDS
RUN apt update
RUN apt install -y wget git && rm -rf /var/lib/apt/lists/*

# INSTALL MINICONDA
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh 

# COPY OVERCOOKED FOLDER
COPY . overcooked-explainability/
WORKDIR overcooked-explainability/

# CREATE CONDA ENVIRONMENT
RUN conda create -n overcooked_env python=3.7

# INSTALL OVERCOOKED PROJECT
RUN conda run -n overcooked_env bash install.sh

# UNCOMMENT IF WE DON'T WANT GIT
# RUN rm -rf .git

# INITIALIZE CONDA
RUN conda init bash
