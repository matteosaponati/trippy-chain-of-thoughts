#!/bin/bash

export CUDA_VISIBLE_DEVICES = 0  

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# python script to run
SCRIPT="finetune-classification-test.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_log_$TIMESTAMP.out"
PID_FILE="$LOG_DIR/train_pid_$TIMESTAMP.pid"

echo "running $SCRIPT on GPU $CUDA_VISIBLE_DEVICES"
echo "logging to $LOG_FILE"
echo "to monitor: tail -f $LOG_FILE"
nohup python "$SCRIPT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Process ID: $(cat "$PID_FILE")"