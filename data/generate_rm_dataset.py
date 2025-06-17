##########
# WIP
##########

# Inference n responses, and create pairwise comparisons for manual labeling

if __name__ == "__main__":
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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

    # charles + gpt:
    # Part 1: Generate 5 Prompts and Create 10 Pairwise Comparisons in a CSV
    import itertools
    import pandas as pd
    from peft import get_peft_model
    import torch

    peft_model = get_peft_model(policy, peft_config)

    text_instr = "You are a math expert with clear and concise reasoning. Solve this problem step-by-step and box your final numerical answer:"
    text_input = "A book with 50 pages, numbered 1 to 50, has its pages renumbered in reverse (page 1 becomes 50, page 2 becomes 49, etc.). How many pages retain the same ones digit before and after renumbering?"
    text_inference = text_instr + "\n" + text_input

    # Generate completions
    inputs = tokenizer(text_inference, return_tensors="pt").to(peft_model.device)
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=2,
    )

    # Save completions
    # Figure out how many tokens were used for the prompt:
    prompt_length = inputs["input_ids"].shape[1]

    # Decode only tokens beyond the prompt
    completions = []
    for output in outputs:
        # Slice off the prompt tokens to keep only the model’s response
        response_tokens = output[prompt_length:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        completions.append(response_text)
    for i, completion in enumerate(completions):  # Print completions and their scores
        print(f"\n--- Completion {i + 1} ---")
        print(completion)

    # Save completions with prompt (to be used by score_completions.py)
    import pandas as pd
    df = pd.DataFrame({
        "prompt": [text_inference] * len(completions),
        "completion": completions,
    })
    csv_path = "/content/drive/MyDrive/Colab_Notebooks/completions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Completions saved to: {csv_path}")

    # Create pairwise comparisons
    pairs = list(itertools.combinations(range(len(completions)), 2))

    data = []
    for i, j in pairs:
        row = {
            "system": text_instr,
            "prompt": text_input,
            "A_index": i,
            "A_response": completions[i],
            "B_index": j,
            "B_response": completions[j],
            "preference": ""  # leave blank to fill manually
        }
        data.append(row)

    df = pd.DataFrame(data)
    csv_path = "/content/drive/MyDrive/Colab_Notebooks/my_dataset/pairwise_comparisons.csv"
    df.to_csv(csv_path, index=False)
    print("CSV saved: pairwise_comparisons.csv")