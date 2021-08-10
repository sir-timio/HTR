# MTS
MTS summer school homeworks

Handwritten text recognition on line level
used dataset: https://github.com/abdoelsayed2016/HKR_Dataset

# HOW TO USE WITH DOCKER
- install DOCKER on your system
- open terminal and move to this directory
- type in terminal (this will take several minutes):
```bash
docker build -t mts/test .
```
- than type in terminal (change path to your local path to MTS repo):
```bash
docker run -it -p 8888:8888 -v /path/to/MTS/homework/repository:/home/mts mts/test
```
- now you can use it like linux

- to run script type:
```bash
conda activate mts && python my_app.py
```
- To open `jupyter notebook` type:
```bash
conda activate mts && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```

- Then copy link to browser and enjoy!

