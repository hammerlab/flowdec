FROM tensorflow/tensorflow:1.6.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends git vim

RUN mkdir /repos && cd /repos && \
    git clone https://github.com/hammerlab/flowdec.git && \
    cd flowdec/python && \
    pip install .

RUN mkdir /notebooks/flowdec

COPY python/examples /notebooks/flowdec
