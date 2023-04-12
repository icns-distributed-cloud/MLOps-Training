FROM nvcr.io/nvidia/pytorch:21.10-py3

LABEL maintainer="inhun321@khu.ac.kr"

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt
RUN pip install --no-cache -U torch torchvision numpy Pillow
RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app
COPY . /usr/src/app

ENTRYPOINT [ "python3", "main.py" ]