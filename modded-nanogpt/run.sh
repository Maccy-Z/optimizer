torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) ./train_gpt_simple.py
