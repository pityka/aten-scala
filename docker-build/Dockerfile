FROM ubuntu
RUN apt-get update 
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch==1.5
RUN echo '/usr/local/lib/python3.8/dist-packages/torch/lib/' > /etc/ld.so.conf && ldconfig -v
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-8-jdk
RUN apt-get install -y clang