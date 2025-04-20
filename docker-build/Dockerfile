FROM pityka/base-ubuntu-libtorch:torch260_amd64
RUN apt-get update --fix-missing && apt-get install -y clang
RUN apt-get update --fix-missing && apt-get install -y wget
RUN apt-get update --fix-missing && apt-get install -y software-properties-common


RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&  dpkg -i cuda-keyring_1.1-1_all.deb &&  apt-get update && apt-get -y install cuda-toolkit-12-4
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-17-jdk-headless
RUN wget -O /usr/local/bin/sbt https://raw.githubusercontent.com/sbt/sbt/5bd44058ef6a96171cfa7cb2779b725fa40633a7/sbt  && chmod u+x /usr/local/bin/sbt && sbt -sbt-create -Dsbt.rootdir=true --version

ENV LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib/