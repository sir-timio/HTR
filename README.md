# MTS
MTS summer school homeworks

Handwritten text recognition on line level
used dataset: https://github.com/abdoelsayed2016/HKR_Dataset

# HOW TO USE WITH DOCKER
- install DOCKER on your system
- open terminal and move to this directory
- type in terminal (this will take several minutes):
```bash
docker build -t mts/tfgpu .
```
- than type in terminal (change path to your local absolut path to MTS repo):
```bash
docker run -it -p 8888:8888 -v /absolut/path/to/MTS/homework:/home/mts mts/tfgpu
```
- Then copy link to browser and enjoy BEST TEXT RECOGNITION NEURAL NETWORK in jupyter!

