FROM pityka/base-ubuntu-libtorch:torch190
RUN apt-get install -y clang
RUN apt-get install -y wget
RUN apt-get update --fix-missing && apt-get install -y software-properties-common

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && apt-get update && apt-get -y install cuda-minimal-build-11-3
