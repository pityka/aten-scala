FROM ubuntu:22.04
RUN apt-get update 
RUN apt-get install -y python3 python3-pip
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-17-jre-headless
RUN pip3 install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/torch/lib/
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.7"
ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib/
