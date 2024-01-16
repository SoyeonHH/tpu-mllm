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