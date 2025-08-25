# 🌀 trippy chain-of-thoughts: training an LLM to be a mathematician high on LSD

*What happens when you teach a language model to solve math problems while tripping through surreal dimensions of reasoning?*

This project explores the boundaries of fine-tuning by training small language models to solve mathematical problems using deliberately hallucinogenic, stream-of-consciousness reasoning that somehow still leads to correct answers.

## the experiment

Instead of training models on boring, step-by-step mathematical reasoning, we create datasets where models learn to:
1. **generate surreal, LSD-inspired reasoning** in `<trip_before>` tags
2. **produce the correct numerical answer** in `<answer>` tags  
3. **end with psychedelic stream-of-consciousness** in `<end>` tags

## how to do it

### 1. dataset generation
- use teacher models (GPT-4o-mini or Qwen2.5-7B) to generate trippy rationales for GSM8K problems
- strict formatting with custom tags ensures consistent structure
- high temperature (0.9) and top-p (0.9) for maximum creativity
- **success rates:** GPT-4o-mini (~99.5%), Qwen2.5-7B (~86%)

### 2. supervised fine-tuning 
- **base model:** Qwen2.5-7B-Instruct
- **method:** QLoRA (parameter-efficient fine-tuning)
- **hardware:** RTX 2080 11GB cards
- **batch strategy:** Gradient accumulation (effective batch size 32)

### 3. Results
- **standard reasoning:** ~81.6% accuracy on GSM8K
- **trippy reasoning:** ~18.7% accuracy (surprisingly high for completely surreal logic!)
- models learn to structure reasoning with custom tags
- emergent behaviors on non-mathematical prompts?

## quick start

### generate trippy dataset
```bash
python scripts/data-generation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mode trippy \
    --dataset gsm8k \
    --batch_size 10
```

### finetune the model
```bash
python scripts/ft-step.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --mode trippy \
    --finetuning True \
    --epochs 1
```

## project structure

```bash
trippy-chain-of-thoughts/
├── src/
│   ├── GeneratorModel.py          # dataset generation
│   ├── TrainerTrippyModel.py      # fine-tuning pipeline  
│   ├── filters.py                 # answer validation & parsing
│   ├── prompts/                   # system prompts for different modes
│   └── adapters.py                # dataset adapters
├── scripts/
│   ├── data-generation.py         # generate trippy datasets
│   ├── ft-step.py                 # fine-tuning script
└── README.md
```