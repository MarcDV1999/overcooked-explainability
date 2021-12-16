FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

### Instalem Python i Llibreries varies
# DEBIAN_FRONTEND=noninteractive per a que no ens demani respostes interactives

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip git python-is-python3 curl ffmpeg




### Instalem Anaconda
WORKDIR /tmp

RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b

ENV PATH=/root/anaconda3/bin:${PATH}

RUN . /root/anaconda3/bin/activate

RUN conda init

RUN conda update conda

RUN conda create -n harl python=3.7





### Instalem Overcooked
RUN mkdir /OvercookedCode

WORKDIR /OvercookedCode

COPY install.sh .
COPY README.md .

RUN conda run -n harl --no-capture-output bash install.sh  

WORKDIR /OvercookedCode/human_aware_rl/human_aware_rl


#RUN conda run -n harl --no-capture-output python run_tests.py


WORKDIR /OvercookedCode/human_aware_rl/human_aware_rl

COPY all_experiments.sh .

CMD conda run -n harl --no-capture-output bash all_experiments.sh 



# Si fa un kill esque li hem de permetre mes RAM
