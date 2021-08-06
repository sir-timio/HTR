FROM conda/miniconda3

WORKDIR /home/mts

COPY mts-env.yaml mts-env.yaml
RUN conda env create -f mts-env.yaml

RUN echo "conda activate mts" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init
RUN conda activate mts && \
	jupyter kernelspec remove -f python3 && \
	ipython kernel install --name mts --user && \
	python -m pip install -U \
	https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.5.0-cp39-cp39-manylinux2010_x86_64.whl \
	&& pip install -U imgaug

EXPOSE 8888
# CMD ['/bin/bash', '&&', 'conda', 'activate', '	mts']
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
# jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root