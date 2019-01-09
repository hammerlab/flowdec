FROM tensorflow/tensorflow:1.6.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends git vim

RUN mkdir -p /repos/flowdec

#
# This will build an image based on the current state of your local source.
# With pip install -e, the installed Python package will point back
# at the source directory, /repos/flowdec.  Any changes made in that
# tree will be used without having to install again.
#
# This is most helpful when used in combination with a bind mount of
# your local source tree on top of this location, so you can make edits
# as usual, and not within the running container.
#
# Note that if you change the structure of the project in any way,
# like adding/removing/renaming/moving files, you will either need to
# rebuild the image, or run the "pip install" command again inside the
# container.
#
COPY . /repos/flowdec/.
RUN cd /repos/flowdec/python && pip install -e .

RUN mkdir /notebooks/flowdec

COPY python/examples /notebooks/flowdec
