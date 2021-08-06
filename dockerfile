FROM continuumio/anaconda3

COPY mts-env.yaml mts-env.yaml
RUN conda env create -f mts-env.yaml

# RUN echo "conda activate mts" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV NB_USER mts
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

WORKDIR /home/${NB_USER}
USER $NB_USER

RUN conda activate mts && \
	jupyter kernelspec remove -f python3 && \
	ipython kernel install --name mts --user

EXPOSE 8888
CMD ["conda", "activate", "mts", "&&", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
