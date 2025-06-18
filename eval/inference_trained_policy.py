# Inference using 1.) base model, 2.) sft-trained model, 3.) grpo-trained model

"""
# Charles Colab script
!python eval/inference_trained_policy.py \
  --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
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

import torch
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

import re

def extract_boxed(text):
    """
    Extracts the numerical value inside the first \boxed{} expression in the given string.
    Parameters:
        text (str): The input string containing LaTeX-style boxed expression.
    Returns:
        int or None: The extracted number if found, otherwise None.
    """
    match = re.search(r"\\boxed\{(\d+)\}", text)
    if match:
        return int(match.group(1))
    return None

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

    # load peft config
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    # Create inference input
    from datasets import Dataset
    import pandas as pd

    # load from csv
    # df = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/my_dataset/sft_dataset.csv")
    # df_text = df["text"].to_list()
    # # df = df[["text"]]  # keep only the 'text' column

    # load squad dataset from hf
    df = load_dataset("rajpurkar/squad", split="validation") # split="train" or "validation"
    df_context = df["context"]
    df_question = df["question"]

    # concatenate context + question
    df = df.map(lambda x: {"prompt": x["context"] + " " + x["question"]})
    # df_prompt = df_context + df_question

    # debug prints
    print("First context:")
    print(df_context[0])
    print("First question:")
    print(df_question[0])
    print("First prompt:")
    print(df["prompt"][0])

    # system prompt
    text_instr = "You are a math expert with clear and concise reasoning. Solve this problem step-by-step and box your final numerical answer:"

    # input
    text_input = "What is 2+2?"

    # combined system prompt + input
    text_inference = text_instr + "\n" + text_input + "\n"

    # debug prints
    # print("Manual question:")
    # print(text_inference)

    inputs = tokenizer(
        # text_inference, # manual question
        df["prompt"][0],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    # hihi test1

    ################
    # Generate completions before training
    ################

    # Craete fresh peft model (for loading in 8-bit)
    from peft import get_peft_model
    import torch
    peft_base = get_peft_model(base_model, peft_config)
    peft_base.eval()
    inputs.to(peft_base.device) # Create fresh peft model
    # model_device = next(peft_base.parameters()).device
    # inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # print("Batch size:", inputs["input_ids"].shape[0]) # debug: batch generate index out of range

    outputs = peft_base.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        # temperature=0.7,
        # num_return_sequences=2,
    )

    # Figure out how many tokens were used for the prompt:
    prompt_length = inputs["input_ids"].shape[1]
    print(f"prompt_length = {prompt_length}")

    # Decode only tokens beyond the prompt
    completions = []
    for output in outputs:
        # Slice off the prompt tokens to keep only the model’s response
        response_tokens = output[prompt_length:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        completions.append(response_text)
    print("\nBase Model Inference:")
    for i, completion in enumerate(completions):  # Print completions and extract boxed answer
        print(f"\n--- Completion {i + 1} ---")
        print(completion)

    ################
    # Generate completions after sft training
    ################

    # Load sft-trained peft model
    from peft import PeftModel
    adapter_path = "/content/drive/MyDrive/Colab_Notebooks/gpt2-sft"
    peft_sft = PeftModel.from_pretrained(base_model, adapter_path)  # Load peft model
    peft_sft.eval()
    inputs.to(peft_sft.device)

    outputs = peft_sft.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        # temperature=0.7,
        # num_return_sequences=2,
    )

    # Figure out how many tokens were used for the prompt:
    prompt_length = inputs["input_ids"].shape[1]

    # Decode only tokens beyond the prompt
    completions = []
    for output in outputs:
        # Slice off the prompt tokens to keep only the model’s response
        response_tokens = output[prompt_length:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        completions.append(response_text)
    print("\nSFT Model Inference:")
    for i, completion in enumerate(completions):  # Print completions and their scores
        print(f"\n--- Completion {i + 1} ---")
        print(completion)
        print(f"Prediction = {extract_boxed(completion)}")
        print(f"Answer = {df['answer'][i]}")

    # ################
    # # Generate completions after grpo training
    # ################
    #
    # # Load grpo-trained peft model
    # from peft import PeftModel
    # adapter_path = "/content/drive/MyDrive/Colab_Notebooks/llama-1B-grpo"
    # peft_grpo = PeftModel.from_pretrained(base_model, adapter_path) # Load peft model
    # peft_grpo.eval()
    # inputs.to(peft_grpo.device)
    #
    # outputs = peft_grpo.generate(
    #     **inputs,
    #     max_new_tokens=1024,
    #     do_sample=False,
    #     # temperature=0.7,
    #     # num_return_sequences=2,
    # )
    #
    # # Figure out how many tokens were used for the prompt:
    # prompt_length = inputs["input_ids"].shape[1]
    #
    # # Decode only tokens beyond the prompt
    # completions = []
    # for output in outputs:
    #     # Slice off the prompt tokens to keep only the model’s response
    #     response_tokens = output[prompt_length:]
    #     response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    #     completions.append(response_text)
    # print("\nGRPO Model Inference:")
    # for i, completion in enumerate(completions):  # Print completions and their scores
    #     print(f"\n--- Completion {i + 1} ---")
    #     print(completion)