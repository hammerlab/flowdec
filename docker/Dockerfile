FROM tensorflow/tensorflow:1.6.0-py3

ENV TF_GPU false

RUN apt-get update && apt-get install -y --no-install-recommends git vim

RUN mkdir /repos && cd /repos && \
    git clone https://github.com/eric-czech/flowdec.git && \
    cd flowdec/python && \
    pip install .

RUN mkdir /notebooks/flowdec

COPY python/examples /notebooks/flowdec