FROM pityka/base-ubuntu-libtorch:torch251
RUN apt-get update --fix-missing && apt-get install -y wget curl
RUN wget -O /usr/local/bin/sbt https://raw.githubusercontent.com/sbt/sbt/5bd44058ef6a96171cfa7cb2779b725fa40633a7/sbt  && chmod u+x /usr/local/bin/sbt && sbt -sbt-create -Dsbt.rootdir=true --version