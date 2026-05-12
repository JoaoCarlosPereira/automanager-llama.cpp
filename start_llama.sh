#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

MODEL_PATH="/media/docker/models/gemma4/gemma-4-31B-it-UD-Q3_K_XL.gguf"

# Inicia o servidor com máxima performance nas GPUs
# -ngl 99: Offload de todas as camadas para GPU
# Usando a porta 8085
# Adicionado --no-mmap para forçar carregamento em RAM/VRAM se necessário, mas mantendo default para velocidade
llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    --flash-attn on \
    --host 0.0.0.0 \
    --port 8085 \
    --tools all \
    --parallel 4 \
    --ctx-size 32768 \
    --mlock
