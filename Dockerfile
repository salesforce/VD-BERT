FROM  pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# install dependencies
RUN apt update -y
RUN apt install wget vim zip unzip ca-certificates-java openjdk-8-jdk openssh-server tmux openssh-client -y

WORKDIR /export/share/yuewang/VD-BERT-Clean

# prepare environment
RUN source /export/share/yuewang/VD-BERT-Clean/.bashrc

EXPOSE 8888