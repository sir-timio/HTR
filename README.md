# MTS
MTS summer school homeworks

Handwritten text recognition on line level


# HOW TO USE WITH DOCKER
- install DOCKER on your system
- open terminal and move to this directory
- type in terminal (this will take several minutes):
```bash
docker build -t mts/test .
```
- than type in terminal:
```bash
docker run -it -p 8888:8888 -v /mnt/d/python/MTS:/home/mts/mnt mts/test
```
- now you can use it like linux
- To open `jupyter notebook` type:
```bash
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```