FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN apt update -y && apt install -y git libgl1-mesa-glx htop
RUN pip install nvitop

WORKDIR /d1/perlin/library/permutation-learning/
COPY requirements_docker.txt requirements_docker.txt
RUN pip install -r requirements_docker.txt
COPY . .

WORKDIR /d1/perlin/library
RUN git clone https://github.com/idiap/fast-transformers.git
WORKDIR /d1/perlin/library/fast-transformers
RUN python setup.py build | tee build.log
RUN python setup.py install | tee install.log

WORKDIR /d1/perlin/library/permutation-learning/