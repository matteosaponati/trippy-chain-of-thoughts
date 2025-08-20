#!/bin/bash

## activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trippy-cot

## huggingface login
echo "HuggingFace access token:"
read -s HUGGING_FACE_HUB_TOKEN
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "no token entered, exiting."
    exit 1
fi
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"

## set GPU
GPU="${GPU:-1}"                                    
export CUDA_VISIBLE_DEVICES="$GPU"
## set paraameters
# MODEL="${MODEL:-Qwen/Qwen2.5-7B}"                  
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"                  
MODE="${MODE:-evaluate}"                  
ARGS=(--model "$MODEL" --mode "$MODE")

# make logs and run python script
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

SCRIPT="teacher.py"   

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/generate_log_${TIMESTAMP}.out"
PID_FILE="$LOG_DIR/generate_pid_${TIMESTAMP}.pid"

echo "running $SCRIPT on GPU $CUDA_VISIBLE_DEVICES"
echo "to monitor: tail -f $LOG_FILE"
nohup python "$SCRIPT" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Process ID: $(cat "$PID_FILE")"