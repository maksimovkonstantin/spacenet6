FROM catalystteam/catalyst:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget htop mc
COPY ./requirements.txt /tmp
RUN conda install -y gdal
RUN pip install -r /tmp/requirements.txt