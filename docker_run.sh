sudo docker run -it --name tpu-torch \
    -d --privileged \
    -p 7860:7860 \
    -v `pwd`:/workspace \
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm \
    /bin/bash