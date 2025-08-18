#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trippy-cot

echo "HuggingFace access token:"
read -s HUGGING_FACE_HUB_TOKEN
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "no token entered, exiting."
    exit 1
fi
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"

GPU="${GPU:-2}"                                    
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"                  
n="${n:-3}"                                        
MAX_NEW="${MAX_NEW:-256}"
TEMP="${TEMP:-0.5}"
TOP_P="${TOP_P:-0.9}"                        
LOAD8BIT="${LOAD8BIT:-1}"                                             
DTYPE="${DTYPE:-auto}"                              
STOP_SEQS="${STOP_SEQS:-}"                          
ARGS=( --model "$MODEL"
       --n "$n"
       --max_new_tokens "$MAX_NEW"
       --temperature "$TEMP"
       --top_p "$TOP_P"
       --load_in_8bit "$LOAD8BIT"
       --torch_dtype "$DTYPE" )

export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

# python script to run
SCRIPT="generate_data.py"   

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/generate_log_${TIMESTAMP}.out"
PID_FILE="$LOG_DIR/generate_pid_${TIMESTAMP}.pid"

echo "running $SCRIPT on GPU $CUDA_VISIBLE_DEVICES"
echo "logging to $LOG_FILE"
echo "to monitor: tail -f $LOG_FILE"
nohup python "$SCRIPT" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Process ID: $(cat "$PID_FILE")"