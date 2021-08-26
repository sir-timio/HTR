FROM  tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update && \
	apt-get install -y libgl1
# libgl1 - library for opencv

WORKDIR /home/htr
COPY requirements.txt .

RUN python -m pip install -r requirements.txt

CMD ["flask", "run", "--host=0.0.0.0"]  
