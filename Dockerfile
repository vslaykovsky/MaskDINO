FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel


WORKDIR /app/

COPY requirements.txt /app/
RUN apt-get update && apt-get install -y git build-essential libglib2.0-0 libgl1
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install -r requirements.txt

COPY maskdino /app/maskdino
COPY configs /app/configs
COPY datasets/dataset_coco_e10 /app/datasets/dataset_coco_e10
COPY entrypoint.sh train_net.py /app/

RUN apt-get install -y curl
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-439.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-439.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh

COPY run.ipynb /app/

ENTRYPOINT ["./entrypoint.sh"]