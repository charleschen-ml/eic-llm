# Inference using 1.) base model, 2.) sft-trained model

"""
# Charles Colab script
!python eval/inference_trained_policy.py \
  --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
  --model_name_or_path openai-community/gpt2 \
  --reward_model_path meta-llama/Llama-3.2-1B-Instruct \
  --sft_model_path meta-llama/Llama-3.2-1B-Instruct \
  --use_peft \
  --load_in_8bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
    # dataset_name: not used (defined in utils.py)
    # model_name_or_path: tokenizer model, base model
    # reward_model_path: not used (defined in ppo_config.py)
    # sft_model_path: policy model
"""

import shutil
import evaluate
import json
from tqdm import tqdm
import torch
from peft import get_peft_model
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# Paths
eval_json_path = "/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json"

# Load validation examples from JSON
with open(eval_json_path, "r") as f:
    dataset = [json.loads(line) for line in f][:2]

# Load SQuAD metric
metric = evaluate.load("squad")

# Score squad metrics (EM, F1) after inference
def score_squad(predictions, references):
    """Compute and print SQuAD EM and F1 scores."""
    metric = evaluate.load("squad")
    results = metric.compute(predictions=predictions, references=references)

    print(f"Exact Match: {results['exact_match']:.2f}")
    print(f"F1 Score: {results['f1']:.2f}")
    num_correct = int(results["exact_match"] * len(predictions) / 100)
    print(f"{num_correct} out of {len(predictions)} predictions were exact matches.")
    return results

if __name__ == "__main__":
    # parse script arguments
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # Set seed for reproducibility
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # Enforce determinism

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    print(f"Loaded base model path: {model_args.model_name_or_path}")

    # load peft config
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    # load squad dataset from hf
    df = load_dataset("rajpurkar/squad", split="train") # split="train" or "validation"
    df_context = df["context"]
    df_question = df["question"]
    df = df.map(lambda x: {"prompt": x["context"].strip() + "\n" + x["question"].strip() + "\n"}) # match sft style

    ################
    # Generate completions before training
    ################

    # Craete fresh peft model (for loading in 8-bit)
    peft_base = get_peft_model(base_model, peft_config)
    peft_base.eval()

    # Inference loop
    predictions, references = [], []

    for example in tqdm(dataset, desc="Evaluating", disable=True):
        context = example["context"].strip()
        question = example["question"].strip()
        qid = example.get("id", f"id_{len(predictions)}")
        prompt = f"{example['context'].strip()}\n{example['question'].strip()}"
        print(f"prompt = \n{prompt}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(peft_base.device)

        with torch.no_grad():
            outputs = peft_base.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        predictions.append({
            "id": qid,
            "prediction_text": generated
        })

        references.append({
            "id": qid,
            "answers": example["answers"]
        })

    print(f"predictions = \n{predictions}")
    print(f"references = \n{references}")
    results = score_squad(predictions, references)

    # ################
    # # Generate completions after sft training
    # ################
    #
    # # Load sft-trained peft model
    # from peft import PeftModel
    # adapter_path = "/content/drive/MyDrive/Colab_Notebooks/gpt2-sft"
    # peft_sft = PeftModel.from_pretrained(base_model, adapter_path)  # Load peft model
    # peft_sft.eval()
    # inputs.to(peft_sft.device)
    #
    # outputs = peft_sft.generate(
    #     **inputs,
    #     max_new_tokens=512,
    #     do_sample=False,
    #     # temperature=0.7,
    #     # num_return_sequences=2,
    # )
    #
    # # Figure out how many tokens were used for the prompt:
    # prompt_length = inputs["input_ids"].shape[1]
    # print(f"prompt_length = {prompt_length}")
    #
    # # Decode only tokens beyond the prompt
    # completions = []
    # for output in outputs:
    #     # Slice off the prompt tokens to keep only the model’s response
    #     response_tokens = output[prompt_length:]
    #     response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    #     completions.append(response_text)
    # print("\nSFT Model Inference:")
    # for i, completion in enumerate(completions):  # Print completions and their scores
    #     print(f"\n--- Completion {i + 1} ---")
    #     print(completion)
    #     # print(f"Prediction = {extract_boxed(completion)}")
    #     # print(f"Answer = {df['answer'][i]}")