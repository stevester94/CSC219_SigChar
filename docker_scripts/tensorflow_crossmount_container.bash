docker run -it --name="Hephaestus" -u $(id -u):$(id -g) -v $(realpath ..):/cross_mount  --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 bash

