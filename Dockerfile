FROM catalystteam/catalyst:20.03-fp16
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget htop mc
COPY ./requirements.txt /tmp
RUN conda install -r /tmp/requirements.txt