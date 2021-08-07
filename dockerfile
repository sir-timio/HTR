FROM conda/miniconda3
WORKDIR /home/mts

COPY mts-env.yaml mts-env.yaml
RUN conda env create -f mts-env.yaml

SHELL ["/bin/bash", "--login", "-c"]

RUN conda init
RUN conda activate mts && \
	apt-get update && \
	apt-get install -y libgl1 && \
	python -m pip install -U \
	https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.5.0-cp39-cp39-manylinux2010_x86_64.whl && \
	pip install -U imgaug && \
	ipython kernel install --name mts --user

SHELL ["conda", "run", "-n", "mts", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mts", "/bin/bash", "-c"]