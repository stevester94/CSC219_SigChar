docker run -it -u $(id -u):$(id -g) -v $(realpath ..):/cross_mount  --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 bash

