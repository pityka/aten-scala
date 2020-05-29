FROM ubuntu
RUN apt-get update 
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch==1.5
RUN echo '/usr/local/lib/python3.8/dist-packages/torch/lib/' > /etc/ld.so.conf && ldconfig -v
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y openjdk-8-jdk
RUN apt-get install -y curl
RUN echo "deb https://dl.bintray.com/sbt/debian /" |  tee -a /etc/apt/sources.list.d/sbt.list && curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add && apt-get update && apt-get install -y sbt
RUN sbt -batch exit
