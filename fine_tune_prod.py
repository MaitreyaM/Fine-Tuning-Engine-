import torch, os, argparse
import bitsandbytes as bnb
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_sharegpt
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def run_sft(args):
    print("--- Starting Production SFT Run ---")
    print(f"Model: {args.model_name}, Dataset: {args.dataset_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
        load_in_4bit=True, token=os.environ.get("HUGGING_FACE_TOKEN"),
    )

    lora_target_modules = find_all_linear_names(model)
    print(f"✅ Found {len(lora_target_modules)} target modules for LoRA: {lora_target_modules}")
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, target_modules=lora_target_modules, lora_alpha=args.lora_alpha,
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    # --- Data Preparation Pipeline ---
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if "hinglish-everyday-conversations" in args.dataset_name.lower():
        print("Applying specific Hinglish Conversational formatting...")
        def format_hinglish(example):
            return {"conversations": [{"role": "user", "content": example["input"]}, {"role": "assistant", "content": example["output"]}]}
        dataset = dataset.map(format_hinglish)
    else:
        dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    dataset = dataset.map(formatting_prompts_func, batched=True)

    train_data = dataset
    eval_data = None
    if args.eval_split_size > 0:
        dataset_split = dataset.train_test_split(test_size=args.eval_split_size, seed=args.seed)
        train_data = dataset_split["train"]
        eval_data = dataset_split["test"]

    output_dir = os.path.join(args.base_path, args.output_dir)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_data, eval_dataset=eval_data,
        dataset_text_field="text", max_seq_length=args.max_seq_length, packing=False,
        args=TrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps, warmup_steps=args.warmup_steps,
            max_steps=args.max_steps, learning_rate=args.learning_rate, fp16=True, bf16=False,
            logging_steps=10, optim="adamw_8bit", eval_strategy="steps" if eval_data is not None else "no",
            eval_steps=args.save_steps if eval_data is not None else None, save_strategy="steps",
            save_steps=args.save_steps, save_total_limit=3, load_best_model_at_end=eval_data is not None,
            report_to="none",
        ),
    )

    # --- THE FINAL, CORRECTED FIX ---
    # We use the official helper with the `force_match=False` parameter,
    # which disables the new, buggy matching logic.
    if args.chat_template:
        print("✅ Applying response-only training with `force_match=False`.")
        trainer = train_on_responses_only(
            trainer,
            # We still need to provide some parts, even if force_match is False.
            # We will use the most common parts as a robust default.
            instruction_part = "user\n",
            response_part = "assistant\n",
            force_match = False,
        )
    # --- END OF FIX ---

    print("Training started...")
    trainer.train()
    print("Training complete.")

    print("Saving final merged model...")
    trainer.model.save_pretrained_merged(f"{output_dir}/final_merged_model", tokenizer, save_method="merged_16bit")
    print(f"Final merged model saved to: {output_dir}/final_merged_model")

if __name__ == '__main__':
    # The argparse section remains the same
    parser = argparse.ArgumentParser(description="The Final, Universal SFT Engine")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="mlabonne/FineTome-100k")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split_size", type=float, default=0.05)
    parser.add_argument("--chat_template", type=str, default="chatml")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_run")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    run_sft(args)