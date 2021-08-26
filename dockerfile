FROM  tensorflow/tensorflow:2.4.1-gpu

WORKDIR /home/htr

RUN apt-get update && \
	apt-get install -y libgl1 git unzip wget && \
	wget https://github.com/sir-timio/HTR/archive/refs/heads/server.zip && \
	unzip server.zip && \
	mv HTR-server/* ./ && \
	python -m pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]  
