FROM continuumio/miniconda3
RUN apt-get update && \
	apt-get upgrade -y && \

	apt-get install -y libgl1 vim
# libgl1 - library for opencv

WORKDIR /home/mts

COPY mts-env.yaml .
RUN conda init bash && \
    . ~/.bashrc && \
    conda env create -f mts-env.yaml
