FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

### Instalem Python, ...

# ffmpeg Necesita que li passis el pais. Com ho fem?
RUN apt-get update && apt-get install -y python3 python3-pip git python-is-python3 curl ffmpeg



### Instalem Anaconda
WORKDIR /tmp

# No trobo manera de que no faci falta posar 'y'
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

RUN bash Anaconda3-2020.02-Linux-x86_64.sh

RUN eval "$(/root/anaconda3/bin/conda shell.bash hook)"

RUN conda create -n harl python=3.7

RUN conda activate hard


### Instalem la resta
RUN mkdir /code

WORKDIR /code

RUN bash install.sh

# Nose perque arriba un moment que fa un kill
