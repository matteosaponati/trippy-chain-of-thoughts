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
GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="$GPU"

## set parameters for the Python script
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
TEACHER_NAME="${TEACHER_NAME:-meta-llama/Meta-Llama-3.2-3B-Instruct}"
MODE="${MODE:-default}"
NOTE="${NOTE:-22-august}"

FINETUNING="${FINETUNING:-True}"       # run training step
TESTING="${TESTING:-False}"            # run eval on test set
INFERENCE="${INFERENCE:-False}"        # (present in parser; not used in main yet)

# optional training hyperparams
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"

## optional generation / inference params
LOAD_ADAPTER="${LOAD_ADAPTER:-False}"    
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}" 
N_VOTES="${N_VOTES:-3}"

ARGS=(
  --model_name "$MODEL_NAME"
  --dataset_name "$DATASET_NAME"
  --teacher_name "$TEACHER_NAME"
  --mode "$MODE"
  --note "$NOTE"
  --seq_length "$SEQ_LENGTH"
  --epochs "$EPOCHS"
  --lr "$LR"
  --per_device_bs "$PER_DEVICE_BS"
  --grad_accum "$GRAD_ACCUM"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --save_steps "$SAVE_STEPS"
  --logging_steps "$LOGGING_STEPS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --batch_size "$BATCH_SIZE"
  --n "$N_VOTES"
)

[[ "$FINETUNING" == "False" ]] && ARGS+=(--finetuning)
[[ "$TESTING" == "True" ]] && ARGS+=(--testing)
[[ "$INFERENCE" == "False" ]] && ARGS+=(--inference)
[[ "$LOAD_ADAPTER" == "True" ]] && ARGS+=(--load_adapter)

# make logs and run python script
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

# You can override SCRIPT via environment; defaults to the file containing your provided code
SCRIPT="${SCRIPT:-ft-step.py}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/generate_log_${TIMESTAMP}.out"
PID_FILE="$LOG_DIR/generate_pid_${TIMESTAMP}.pid"

echo "running $SCRIPT on GPU $CUDA_VISIBLE_DEVICES"
echo "to monitor: tail -f $LOG_FILE"
nohup python "$SCRIPT" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Process ID: $(cat "$PID_FILE")"