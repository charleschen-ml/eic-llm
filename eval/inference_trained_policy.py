# Inference using 1.) base model, 2.) sft-trained model,

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
import csv
from tqdm import tqdm
import torch
from peft import get_peft_model, PeftModel
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
# Ensure inference quantization config matches that of QAT
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from qat import (
    patch_linear_forward_with_switchable_quantization,
    set_active_bitwidths,
    add_bitwise_lora_adapters
)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # To fix torch deterministic error
torch.use_deterministic_algorithms(True)

# Paths
eval_json_path = "/content/drive/MyDrive/Colab_Notebooks/eic_llm/train_set.json" #  eval_set.json (regular) or train_set.json (overfit)
adapter_path = "/content/drive/MyDrive/Colab_Notebooks/gpt2-qat" # gpt2-qat or gpt2-sft
OUTPUT_CSV_PATH = "/content/drive/MyDrive/Colab_Notebooks/eic_llm/inference_output.csv" # inference output
bitwise_lora_adapter_path = "/content/drive/MyDrive/Colab_Notebooks/gpt2-qat/full_qat_model.pt"

# Settings
USE_QUANTIZATION = True
USE_BITWISE_LORA = True
BIT_CHOICES = [8, 16] # bit choices for LoRA. Needs to match training/qat.py

# Inference bit config
config1 = {f"transformer.h.{i}": 4 if i % 2 == 0 else 8 for i in range(12)}  # for 12 layers
config2 = {f"transformer.h.{i}": 4 for i in range(12)}
config3 = {f"transformer.h.{i}": 8 for i in range(12)}
config4 = {f"transformer.h.11": 8}
config5 = {f"transformer.h.11": 4}
config6 = {f"transformer.h.6": 8}
config7 = {f"transformer.h.6": 4}
INF_BIT_CONFIG = config6

# Load validation examples from JSON
with open(eval_json_path, "r") as f:
    dataset = [json.loads(line) for line in f][:100]
print(f"Examples used for inference: {len(dataset)}")

# Load SQuAD metric
metric = evaluate.load("squad")

# Score squad metrics (EM, F1) after inference
def score_squad(predictions, references):
    for i in range(min(len(predictions), 2)): # print at most 2 samples
        print(f"prediction {i} = {predictions[i]['prediction_text']}")
        print(f"reference {i} = {references[i]['answers']['text']}")
    metric = evaluate.load("squad")
    results = metric.compute(predictions=predictions, references=references)

    num_correct = int(results["exact_match"] * len(predictions) / 100)
    print(f"Exact Match: {results['exact_match']:.2f} ({num_correct}/{len(predictions)})")
    print(f"F1 Score: {results['f1']:.2f}")
    return results

# Save inference outputs to csv
def save_predictions_to_csv(predictions, references):
    metric = evaluate.load("squad")
    rows = []

    for pred, ref in zip(predictions, references):
        pred_text = pred["prediction_text"]
        ref_texts = ref["answers"]["text"]
        joined_refs = ", ".join([r.strip() for r in ref_texts])

        score = metric.compute(
            predictions=[{"prediction_text": pred_text, "id": pred["id"]}],
            references=[{"answers": {"text": ref_texts, "answer_start": [0] * len(ref_texts)}, "id": ref["id"]}]
        )

        rows.append({
            "prediction": pred_text,
            "reference": joined_refs,
            "exact_match": score["exact_match"],
            "f1_score": score["f1"]
        })
        rows.sort(key=lambda x: x["f1_score"])  # sort by worst predictions

    with open(OUTPUT_CSV_PATH, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prediction", "reference", "exact_match", "f1_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} results to {OUTPUT_CSV_PATH}")

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
    ).to("cuda")
    print(f"Loaded base model path: {model_args.model_name_or_path}")

    # Set quantization config to match training
    if USE_QUANTIZATION:
        patch_linear_forward_with_switchable_quantization(base_model, bit_widths=BIT_CHOICES)
        add_bitwise_lora_adapters(base_model, bit_widths=BIT_CHOICES)

        # Dummy forward to create LoRA modules, which are created at run time to fix matrix dim mismatch error
        dummy_input = tokenizer("hello", return_tensors="pt").to(base_model.device)
        _ = base_model(**dummy_input)

        state_dict = torch.load(bitwise_lora_adapter_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
        base_model.to("cuda")
        set_active_bitwidths(base_model, INF_BIT_CONFIG)
        base_model.eval()

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

    # ################
    # # Generate completions before training
    # ################

    # # Craete fresh peft model (for loading in 8-bit)
    # peft_base = get_peft_model(base_model, peft_config)
    # peft_base.eval()

    # # Inference loop
    # predictions, references = [], []

    # print("\nBEFORE TRAINING:\n")

    # for example in tqdm(dataset, desc="Evaluating", disable=True):
    #     context = example["context"].strip()
    #     question = example["question"].strip()
    #     qid = example.get("id", f"id_{len(predictions)}")
    #     prompt = f"{example['context'].strip()}\n{example['question'].strip()}"
    #     print(f"prompt = \n{prompt}")

    #     inputs = tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512,
    #     ).to(peft_base.device)

    #     with torch.no_grad():
    #         outputs = peft_base.generate(
    #             **inputs,
    #             max_new_tokens=32,
    #             do_sample=False,
    #             eos_token_id=tokenizer.eos_token_id,
    #             pad_token_id=tokenizer.eos_token_id
    #         )

    #     generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    #     generated_truncated = generated.split("\n")[0].strip()

    #     predictions.append({
    #         "id": qid,
    #         "prediction_text": generated_truncated
    #     })

    #     references.append({
    #         "id": qid,
    #         "answers": example["answers"]
    #     })

    # results = score_squad(predictions, references)

    ################
    # Generate completions after sft training
    ################

    # Load sft-trained peft model
    if USE_BITWISE_LORA:
        peft_sft = base_model # use base model for bitwise lora
    else:
        peft_sft = PeftModel.from_pretrained(base_model, adapter_path)  # Load peft model
        peft_sft.eval()

    # Inference loop
    predictions, references = [], []

    print("\nAFTER TRAINING:\n")

    for example in tqdm(dataset, desc="Evaluating", disable=True):
        context = example["context"].strip()
        question = example["question"].strip()
        qid = example.get("id", f"id_{len(predictions)}")
        prompt = f"{example['context'].strip()}\n{example['question'].strip()}\n"
        # print(f"prompt = \n{prompt}")

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(peft_sft.device)

        with torch.no_grad():
            outputs = peft_sft.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        generated_truncated = generated.split("\n")[0].strip()

        predictions.append({
            "id": qid,
            "prediction_text": generated
        })

        references.append({
            "id": qid,
            "answers": example["answers"]
        })

    results = score_squad(predictions, references)
    save_predictions_to_csv(predictions, references)
