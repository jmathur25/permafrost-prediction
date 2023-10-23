FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get -y install wget # libgl-dev

RUN mkdir -p /root/tools

WORKDIR /root/tools

RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

RUN bash Mambaforge-Linux-x86_64.sh -b -p ./mambaforge

RUN ./mambaforge/bin/mamba init bash

RUN ./mambaforge/bin/mamba install wget git tree --yes

COPY conda_envs conda_envs
COPY MintPy/requirements.txt requirements.txt
RUN ./mambaforge/bin/mamba install --file conda_envs/requirements.txt --file requirements.txt isce2">=2.6.3" --yes

# Installs libGL.so.1 which cv2 needs
RUN apt-get update && apt-get install libgl1

RUN echo "source /root/tools/conda_envs/config.rc" >> ~/.bashrc

COPY scripts/init.sh /etc/profile.d/init.sh
RUN chmod +x /etc/profile.d/init.sh

WORKDIR /root
