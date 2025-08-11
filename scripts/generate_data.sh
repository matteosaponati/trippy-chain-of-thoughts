#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trippy-cot

GPU="${GPU:-0}"                                    
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
LIMIT="${LIMIT:-7500}"                             
N="${N:-3}"                                        
MAX_NEW="${MAX_NEW:-384}"
TEMP="${TEMP:-0.9}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-}"                                  
LOAD4BIT="${LOAD4BIT:-1}"                           
CACHE_DIR="${CACHE_DIR:-}"                          
DEVICE="${DEVICE:-}"                                
DTYPE="${DTYPE:-auto}"                              
STOP_SEQS="${STOP_SEQS:-}"                          
ARGS=( --model "$MODEL"
       --limit "$LIMIT"
       --N "$N"
       --max_new_tokens "$MAX_NEW"
       --temperature "$TEMP"
       --top_p "$TOP_P"
       --top_k "$TOP_K"
       --load_in_4bit "$LOAD4BIT"
       --dtype "$DTYPE" )

export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="./logs"
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