
## Server Settings

### 1. Create TPU VM

To create TPU VM in command line:

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export PATH=YOUR_SDK_PATH

cd ~

gcloud config set project ${PROJECT_ID}
gcloud config set account YOUR_EMAIL

export TPU_NAME=tpu-v3-8-01
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=europe-west4-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0
```

### 2. Add an SSH public key to Google Cloud

To view SSH public key:

```bash
cat ~/.ssh/id_rsa.pub
```

### 3. SSH into TPU VM

Create or edit `~/.ssh/config`:

```bash
vi ~/.ssh/config
```

Add content:

```bash
Host tpuv3-8-1
    User sodus1102
    Hostname EXTERNAL_ID
    IdentityFile ~/.ssh/KEY_FILE
```

SSH into the TPU VM using VSCode or command line:

```bash
ssh tpuv3-8-1
```

### 4. Start Docker Container for Pytorch XLA

```bash
sudo docker run -it --name tpu-torch \
    -d --privileged \
    -p 7860:7860 \
    -v `pwd`:/workspace \
    us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm \
    /bin/bash
```

* Pytorch on XLA devices: https://pytorch.org/xla/release/2.1/index.html


### 5. TPU Monitoring (TODO)

* jax-smi: https://github.com/ayaka14732/jax-smi?tab=readme-ov-file


## Code

Explore the code:

```bash
git clone git@github.com:SoyeonHH/tpu-mllm.git
cd tpu-mllm
```


## Datasets

* Raven's IQ Test Dataset: https://aka.ms/kosmos-iq50

```bash
mkdir data && cd data
wget https://aka.ms/kosmos-iq50
```


## MLLMs

* Kosmos-2-patch14-224: https://huggingface.co/microsoft/kosmos-2-patch14-224


## Evaluation

* Phrase grounding task for Kosmos-2: https://github.com/microsoft/unilm/blob/master/kosmos-2/evaluation/flickr_entities/README.md

* Reffering expression comprehension task for Kosmos-2: https://github.com/microsoft/unilm/blob/master/kosmos-2/evaluation/refcoco/README.md

* Reffering expression generation task for Kosmos-2: https://github.com/microsoft/unilm/tree/master/kosmos-2/evaluation

* Image captioning for Kosmos-2: https://github.com/microsoft/unilm/tree/master/kosmos-2/evaluation

* Visual question answering for Kosmos-2: https://github.com/microsoft/unilm/tree/master/kosmos-2#evaluation


