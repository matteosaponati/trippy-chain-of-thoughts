from peft import LoraConfig
from trl import SFTConfig

def config_LoRA(lora_r, lora_alpha, lora_dropout, bias, task_type, target_modules):

    peft_config = LoraConfig(
            r = lora_r,                        ## the rank of the LoRA update                      
            lora_alpha = lora_alpha,           ## the alpha parameters (tipucally alpha = r or alpha = 2 * r)
            lora_dropout = lora_dropout,       ## dropout probability (tipically 0.1)
            bias = bias,                          ## bias for low-rank linear transformation
            task_type = task_type,                ## task type
            target_modules = target_modules)             ## weight matrices that should be fine-tuned

    return peft_config

def config_SFT(output_dir,
            seq_length,
            packing,
            per_device_train_batch_size,
            grad_accum,
            learning_rate,
            num_train_epochs,
            lr_scheduler_type,
            warmup_ratio,
            logging_steps,
            save_steps,
            save_total_limit,
            bf16,
            fp16,                         
            gradient_checkpointing,
            optim,
            report_to):

    sft_cfg = SFTConfig(
            output_dir = output_dir,
            max_length = seq_length,
            packing = packing,                      
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = grad_accum,
            learning_rate = learning_rate,
            num_train_epochs = num_train_epochs,
            lr_scheduler_type = lr_scheduler_type,
            warmup_ratio = warmup_ratio,
            logging_steps = logging_steps,
            save_steps = save_steps,
            save_total_limit = save_total_limit,
            bf16 = bf16,
            fp16 = fp16,                         
            gradient_checkpointing = gradient_checkpointing,
            optim = optim,
            report_to = report_to)
    
    return sft_cfg


