#! /bin/bash
FROM nvcr.io/nvidia/tritonserver:22.07-py3


# for opencv c library (OpenGL)
RUN apt-get update
RUN apt install -y build-essential cmake unzip
RUN apt install -y libyaml-cpp-dev libgoogle-glog-dev libgflags-dev
RUN apt install -y libjpeg-dev libtiff-dev libpng-dev
RUN apt install -y libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev libavresample-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install -y x264 libx264-dev libfaac-dev libmp3lame-dev libv4l-dev v4l-utils ffmpeg libtheora-dev libvorbis-dev libgtk-3-dev libavutil-dev
RUN apt install -y libatlas-base-dev gfortran libeigen3-dev libhdf5-serial-dev python2.7-dev python3.8-dev


RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install ninja-build

# python packages
RUN pip install --upgrade pip
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install easydict==1.9 lmdb==1.2.1 natsort==7.1.1 
RUN pip install batch_jaro_winkler==0.1.0 hdbscan==0.8.27 pandas==1.1.3
RUN pip install transformers==4.6.1 scikit-learn==0.24.2 kss==2.5.1 sentencepiece==0.1.95

RUN pip install omegaconf==2.0.5 lmdb~=1.3.0 timm~=0.6.5 nltk~=3.7.0 Pillow~=9.2.0
RUN pip install imgaug~=0.4.0 fvcore~=0.1.5.post20220512 ray[tune]~=1.13.0 ax-platform~=0.2.5.1 PyYAML~=6.0.0 tqdm~=4.64.0 requests==2.28.1

RUN pip install editdistance Polygon3 pyclipper scipy matplotlib
RUN pip install opencv-contrib-python==4.5.2.54
RUN pip install tensorboard

# COPY . /workspace

# ENTRYPOINT [ "/bin/bash" ]

