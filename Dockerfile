# Specifies base image and tag
FROM us-docker.pkg.dev/vertex-ai/training/tf-tpu-pod-base-cp38:latest
WORKDIR /root

# Download and install `tensorflow` 2.12 
RUN pip install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.12.0/tensorflow-2.12.0-cp38-cp38-linux_x86_64.whl

# Download and install `libtpu`.
# You must save `libtpu.so` in the '/lib' directory of the container image.
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.6.0/libtpu.so -o /lib/libtpu.so

# Copies the trainer code to the docker image.
COPY trainer/task.py /root/trainer.py

# The base image is setup so that it runs the CMD that you provide.
# You can provide CMD inside the Dockerfile like as follows.
# Alternatively, you can pass it as an `args` value in ContainerSpec:
# (https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#containerspec)
CMD ["python3", "trainer.py"]