from unsloth import FastLanguageModel
import torch

model_name = "Qwen/Qwen3-0.6B-Base" # e.g., "unsloth/llama-3-8b-bnb-4bit"
lora_checkpoint = "/mnt/storage/Models/Qwen3-0.6B-Norwegian/stage1_norwegian_pretrain/checkpoint-2200/" # Directory containing adapter_model.safetensors
merged_model_path = "/mnt/storage/Models/Qwen3-0.6B-Norwegian/norwegianmerged_16bit"

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = torch.float16, # Use float16 or bfloat16
        load_in_4bit = True,
    )

    # Apply the LoRA adapter
    model = FastLanguageModel.from_pretrained(
        model_name = lora_checkpoint,
        max_seq_length = 2048,
        dtype = torch.float16,
        load_in_4bit = True,
    )
    # Apply the LoRA adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_checkpoint,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    model.save_pretrained_merged(merged_model_path, tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf(merged_model_path + "_gguf_f16", tokenizer, quantization_method="f16")
    print("Model loaded successfully")

if __name__ == "__main__":
    main()
