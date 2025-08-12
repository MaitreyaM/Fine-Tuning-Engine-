# ml_scripts/fine_tune_prod.py
import torch, os, argparse
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

# --- Production Technique: Data Formatting as a separate function ---
# This makes the script cleaner and easier to adapt for different dataset structures.
def format_openhermes(dataset: Dataset) -> Dataset:
    """Formats the OpenHermes dataset into a single 'text' column."""
    # The dataset has a 'conversations' column, which is a list of dicts.
    # We'll format it into a chat-like string.
    def to_text(example):
        text = ""
        for turn in example['conversations']:
            role = turn['from']
            value = turn['value']
            text += f"### {role.capitalize()}:\n{value}\n\n"
        return {"text": text}

    return dataset.map(to_text)

def format_alpaca_style(dataset: Dataset) -> Dataset:
    """Formats Alpaca-style datasets (like the Hinglish one) into a single 'text' column."""
    def to_text(example):
        instruction = example["instruction"]
        input_text  = example.get("input", "") # Use .get for safety
        output      = example["output"]
        return {"text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"}

    return dataset.map(to_text)

def format_conversational(dataset: Dataset) -> Dataset:
    """Formats conversational datasets with 'input' and 'output' columns."""
    def to_text(example):
        # We will map 'input' to the Human and 'output' to the Assistant
        human_turn = example["input"]
        assistant_turn = example["output"]
        return {"text": f"### Human:\n{human_turn}\n\n### Assistant:\n{assistant_turn}"}

    return dataset.map(to_text)

def run_sft(args):
    """
    Main function to run the production-grade SFT process.
    """
    print("--- Starting Production SFT Run ---")
    print(f"Model: {args.model_name}, Dataset: {args.dataset_name}")

    # 1. Load the dataset and format it
    # Note: This formatting function is specific to OpenHermes.
    # For Alpaca, you would need a different function.
    # For simplicity in this final script, we assume a 'text' column exists or is created.
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    

    # In the run_sft function...

    # --- THIS IS THE FINAL, CORRECT LOGIC ---
    if "hinglish-everyday-conversations" in args.dataset_name.lower():
        print("Applying Conversational formatting...")
        dataset = format_conversational(dataset)
    elif "alpaca" in args.dataset_name.lower() or "hinglish-top-10000" in args.dataset_name.lower():
        print("Applying Alpaca-style formatting...")
        dataset = format_alpaca_style(dataset)
    elif "openhermes" in args.dataset_name.lower():
        print("Applying OpenHermes-style formatting...")
        dataset = format_openhermes(dataset)
    else:
        # If no specific format is known, assume a 'text' column already exists
        print("Warning: Unknown dataset format. Assuming a 'text' column exists.")
    # --- END OF FINAL LOGIC ---

    # Create a proper evaluation split
    dataset_split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_data = dataset_split["train"]
    eval_data = dataset_split["test"]

    # 2. Load the model using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # 3. Configure and apply LoRA adapters
    # NEW: Dynamically select target modules based on model type
    if "deepseek" in args.model_name.lower():
        print("Applying DeepSeek-specific LoRA target modules.")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else: # Default to Llama-style
        print("Applying Llama-style LoRA target modules.")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules, # Use the dynamic variable
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing = "unsloth", # Reverted to the optimized version
        random_state=args.seed,
    )

    # 4. Configure Training Arguments for Production
    # NEW: Construct the absolute path for the output directory
    output_dir = os.path.join(args.base_path, args.output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir, # Use the new absolute path
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16 = True, # Correct for T4 hardware
        bf16 = False, # Correct for T4 hardware
        logging_steps=10,
        optim="adamw_8bit",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
    )

    # 5. Initialize and run the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        dataset_text_field="text", # Assumes a 'text' column
        max_seq_length=args.max_seq_length,
        # packing is True by default, which is the optimized path
        args=training_args,
    )
    
    print("Training started...")
    trainer.train()
    print("Training complete.")

    # Merge adapters and save the final model
    print("Merging LoRA adapters and saving final model...")
    merged_model = model.merge_and_unload()
    
    final_save_path = f"{output_dir}/final_merged_model"
    merged_model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final merged model saved to: {final_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Production SFT Script")
    
    # NEW: The required base_path for guaranteed saving
    parser.add_argument("--base_path", type=str, required=True, help="Absolute base path for outputs.")

    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    parser.add_argument("--dataset_name", type=str, default="teknium/OpenHermes-2.5")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_run")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=-1) # Default to full training run
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()
    run_sft(args)