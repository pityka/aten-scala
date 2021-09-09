FROM ubuntu:20.04
RUN apt-get update 
RUN apt-get install -y python3 python3-pip
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-8-jdk
RUN apt-get install -y curl
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list && echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list && curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add && apt-get update && apt-get install -y sbt
RUN sbt -Dsbt.rootdir=true -batch exit
RUN pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/torch/lib/
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.1"
