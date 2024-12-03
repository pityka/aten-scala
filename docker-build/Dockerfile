FROM pityka/base-ubuntu-libtorch:torch251
RUN apt-get update --fix-missing && apt-get install -y clang
RUN apt-get update --fix-missing && apt-get install -y wget
RUN apt-get update --fix-missing && apt-get install -y software-properties-common


RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb && apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && apt-get update && apt-get -y install cuda-minimal-build-11-7 libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-17-jdk-headless
RUN wget -O /usr/local/bin/sbt https://raw.githubusercontent.com/sbt/sbt/5bd44058ef6a96171cfa7cb2779b725fa40633a7/sbt  && chmod u+x /usr/local/bin/sbt && sbt -sbt-create -Dsbt.rootdir=true --version

ENV LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib/