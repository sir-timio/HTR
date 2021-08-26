FROM  tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN apt-get update && \
	apt-get install -y libgl1
# libgl1 - library for opencv

WORKDIR /home/htr
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

CMD ["python", "-m", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]  
