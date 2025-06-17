##########
# WIP
##########

# Score completion using reward model

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

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # Set seed for reproducibility
    import random
    import numpy as np
    import torch
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # Enforce determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    # if tokenizer.pad_token is None: # charles
    #     tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # charles: suspect of giving index out of range error
    if tokenizer.pad_token is None: # charles
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) # explicitly set pad_token_id

    print(f"pad_token = {tokenizer.pad_token}") # charles
    print(f"pad_token_id = {tokenizer.pad_token_id}") # charles

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    print("Vocab size:", tokenizer.vocab_size)  # charles debug

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    # charles: Set pad_token_id in model configs explicitly
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id
    policy.config.pad_token_id = tokenizer.pad_token_id
    # charles: resize tokenizer to prevent index out of range error
    reward_model.resize_token_embeddings(len(tokenizer))
    value_model.resize_token_embeddings(len(tokenizer))
    policy.resize_token_embeddings(len(tokenizer))

    print("Value model vocab size:", value_model.config.vocab_size)  # charles debug
    print("Reward model vocab size:", reward_model.config.vocab_size)  # charles debug
    print("Policy model vocab size:", policy.config.vocab_size)  # charles debug

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    # charles inference test
    # from peft import get_peft_model
    # import torch
    # import torch
    # torch.manual_seed(42)
    # peft_model = get_peft_model(policy, peft_config)
    # text_instr = "You are a math expert with clear and concise reasoning. Solve this problem step-by-step and box your final numerical answer:"
    # text_input = "A book with 50 pages, numbered 1 to 50, has its pages renumbered in reverse (page 1 becomes 50, page 2 becomes 49, etc.). How many pages retain the same ones digit before and after renumbering?"
    # text_inference = text_instr + "\n" + text_input
    # inputs = tokenizer(text_inference, return_tensors="pt").to(peft_model.device)
    # outputs = peft_model.generate(
    #     **inputs,
    #     max_new_tokens=512,
    #     do_sample=False,
    #     # temperature=0.7,
    #     # num_return_sequences=5,
    # )
    # for i, output in enumerate(outputs):
    #     decoded = tokenizer.decode(output, skip_special_tokens=True)
    #     print(f"\n--- Assistant Reply {i + 1} ---\n{decoded}")

    #
    # # Generate completions locally
    # # charles + gpt:
    # # Part 1: Generate 5 Prompts and Create 10 Pairwise Comparisons in a CSV
    # import itertools
    # import pandas as pd
    #
    # # Set seed for reproducibility
    # import random
    # import numpy as np
    # import torch
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)  # Enforce determinism
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #
    # # Create peft model
    # from peft import get_peft_model
    # peft_policy = get_peft_model(policy, peft_config)
    #
    # text_instr = "You are a math expert with clear and concise reasoning. Solve this problem step-by-step and box your final numerical answer:"
    # text_input = "A book with 50 pages, numbered 1 to 50, has its pages renumbered in reverse (page 1 becomes 50, page 2 becomes 49, etc.). How many pages retain the same ones digit before and after renumbering?"
    # text_inference = text_instr + "\n" + text_input
    #
    # # Generate completions
    # inputs = tokenizer(text_inference, return_tensors="pt").to(peft_policy.device)
    # outputs = peft_policy.generate(
    #     **inputs,
    #     max_new_tokens=1024,
    #     do_sample=True,
    #     temperature=0.7,
    #     num_return_sequences=2,
    # )
    #
    # # Save completions
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
    #
    # # debug index out of range
    # print("Reward model max position embeddings:", reward_model.config.max_position_embeddings)

    # Load completions from csv
    import pandas as pd
    csv_path = "/content/drive/MyDrive/Colab_Notebooks/completions.csv"
    df = pd.read_csv(csv_path)
    completions = df["completion"].tolist()
    text_inference = df["prompt"].iloc[0]  # Load the shared prompt (optional)
    for i, completion in enumerate(completions):  # Print completions and their scores
        print(f"\n--- Completion {i + 1} ---")
        print(completion)

    max_len = min(reward_model.config.max_position_embeddings, 2048)
    rm_inputs = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    print(rm_inputs["input_ids"].shape) # charles
    # print(rm_inputs["input_ids"]) # charles

    # Score completions before training
    rm_inputs.to(reward_model.device)
    reward_model.eval() # charles
    with torch.no_grad():
        rm_outputs = reward_model(**rm_inputs)
        rm_scores = rm_outputs.logits.squeeze(-1).tolist()
    print("\nBefore training:")
    for i, (text, score) in enumerate(zip(completions, rm_scores)):  # Print completions and their scores
        print(f"\n--- Completion {i + 1} ---")
        # print(text)
        print(f"Reward score: {score:.2f}")

    # Load trained rm
    from peft import PeftModel
    adapter_path = "/content/drive/MyDrive/Colab_Notebooks/llama-1B-Reward-LoRA"
    peft_reward = PeftModel.from_pretrained(reward_model, adapter_path)

    # Score completions after training
    rm_inputs.to(peft_reward.device)
    peft_reward.eval() # charles
    with torch.no_grad():
        rm_outputs = peft_reward(**rm_inputs)
        rm_scores = rm_outputs.logits.squeeze(-1).tolist()
    print("\nAfter training:")
    for i, (text, score) in enumerate(zip(completions, rm_scores)):  # Print completions and their scores
        print(f"\n--- Completion {i + 1} ---")
        # print(text)
        print(f"Reward score: {score:.4f}")