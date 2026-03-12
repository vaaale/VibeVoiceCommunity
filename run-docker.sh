export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
docker run -it --rm --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus '"device=2"' \
  -p 7860:7860 -p 4000:3000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/storage/tmp:/audio \
  -v ./demo:/app/demo \
  -v ./vibevoice:/app/vibevoice \
  -v /mnt/storage/Models/Vibevoice/vibevoice-1.5b:/app/models \
  vibevoice-asr /bin/bash

