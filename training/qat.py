# Copied from TRL as is to be run from Colab

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub

# LoRA
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
"""

import argparse
import torch
import torch.nn as nn
import re

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.models.gpt2.modeling_gpt2 import Conv1D

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# Settings
MAX_DATASET_SIZE = 2  # Total samples (train+validation). Set to >= 2.
USE_QUANTIZATION = True
QUANT_BITS = 8
USE_BITWISE_LORA = True

# Paths
bitwise_lora_adapter_path = "/content/drive/MyDrive/Colab_Notebooks/gpt2-qat/full_qat_model.pt"

def quantize_tensor(tensor, num_bits=4) -> object:
    device = tensor.device # capture tensor device (gpu)
    max_val = tensor.abs().max()
    scale = max_val / (2 ** (num_bits - 1) - 1)
    tensor_quant = torch.round(tensor / scale).clamp(
        min=-(2 ** (num_bits - 1)), max=2 ** (num_bits - 1) - 1
    )
    tensor_dequant = tensor_quant * scale
    return tensor_dequant.to(device) # move tensor to gpu

def patch_linear_forward_with_switchable_quantization(model, bit_widths=[4, 8, 16]):
    """
    For each nn.Linear layer, store quantized weights for multiple bit-widths
    and use a runtime flag to choose the active one.
    """
    for name, module in model.named_modules():
        if not name.startswith("transformer.h.11."):
            continue  # only quantize transformer.h.0.*
        # if name == "lm_head": # To remove
        #     print(f"[Quantize] {name} | SKIPPED (lm_head)")
        #     continue  # ← Skip quantizing lm_head
        if isinstance(module, (nn.Linear, Conv1D)):
            module._quantized_weights = {}  # e.g., {4: tensor, 8: tensor}

            # Precompute quantized weights
            for b in bit_widths:
                w = module.weight.detach().clone().to(module.weight.device)
                q_w = quantize_tensor(w, num_bits=b)
                mean_diff = (w - q_w).abs().mean().item()
                max_before = w.abs().max().item()
                print(
                    f"[Quantize] {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Max abs weight before: {max_before:.4f}")
                module._quantized_weights[b] = q_w

            module._active_bit = bit_widths[0]  # default
            module._bit_choices = bit_widths

            def quantized_forward(self, input):
                weight = self._quantized_weights[self._active_bit]
                if weight.shape[1] != input.shape[-1]: 
                    weight = weight.T # transpose if dim mismatch (for gpt2 internal layers e.g. c_attn)
                return nn.functional.linear(input, weight, # put bias on the same device as the layer itself
                                            self.bias.to(input.device) if self.bias is not None else None)

            module.forward = quantized_forward.__get__(module, nn.Linear)

def set_active_bitwidths(model, bit_config_dict):
    bit_config_dict = { # debug: only quantize transformer.h.0.*
        "transformer.h.11": 8
    }
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
            layer_id = ".".join(name.split(".")[:3])
            if layer_id in bit_config_dict:
                module._active_bit = bit_config_dict[layer_id]
                print(f"[Quantize] {name} | Matched: {layer_id} | Active bit: {bit_config_dict[layer_id]}")

def add_bitwise_lora_adapters(model, bit_widths=[4, 8, 16]):
    """
    For each Linear layer in transformer.h.0, attach multiple LoRA adapters — one per bit-width.
    During forward pass, apply quantized weight and the matching LoRA adapter.
    """
    for name, module in model.named_modules():
        # Only apply to transformer.h.0.* layers
        if not name.startswith("transformer.h.0."):
            continue

        # Apply only to Linear layers that were quantized
        if isinstance(module, nn.Linear) and hasattr(module, "_quantized_weights"):
            module._lora_adapters = nn.ModuleDict()

            # Create one LoRA module per bit-width (e.g., 4-bit and 8-bit)
            for b in bit_widths:
                r = 8  # LoRA rank; can tune this
                lora_down = nn.Linear(module.in_features, r, bias=False)
                lora_up = nn.Linear(r, module.out_features, bias=False)
                module._lora_adapters[str(b)] = nn.Sequential(lora_down, lora_up)

            # Set default active bit-width
            module._active_bit = bit_widths[0]

            # Patch the forward pass to use selected quantized weight + matching LoRA
            def forward_with_quant_and_lora(self, input):
                # Get quantized weight for current bit-width
                w = self._quantized_weights[self._active_bit]

                # Get the LoRA adapter for current bit-width
                lora = self._lora_adapters[str(self._active_bit)]

                # Transpose if needed for compatibility
                if w.shape[1] != input.shape[-1]:
                    w = w.T

                # Base output + LoRA output
                base_out = nn.functional.linear(input, w, self.bias)
                lora_out = lora(input)
                return base_out + lora_out

            # Replace original forward function
            module.forward = forward_with_quant_and_lora.__get__(module, nn.Linear)

def set_random_bitwidths(model, bit_choices=[4, 8]):
    for name, module in model.named_modules():
        if name.startswith("transformer.h.0") and hasattr(module, "_quantized_weights"):
            module._active_bit = random.choice(bit_choices)

def sft_preprocess(example, tokenizer):
    answer = example["answers"]["text"][0].strip()
    return {
        "text": example["context"].strip() + "\n" + example["question"].strip() + "\n" + answer + tokenizer.eos_token
    }

def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Apply quantization
    if USE_QUANTIZATION:
        model.to("cuda")  # ✅ move to GPU before quantizing
        ### debug
        # print("\n🔍 Listing all Linear layers in model:\n")
        # for name, module in model.named_modules():
        #     if isinstance(module, (nn.Linear, Conv1D)):
        #         print(f"- {name}")
        ### debug
        print("Before patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        patch_linear_forward_with_switchable_quantization(model, bit_widths=[4, 8, 16])
        print("After patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        print(f"⚡ Quantization enabled: using {QUANT_BITS}-bit weight quantization in linear layers.")
        add_bitwise_lora_adapters(model, bit_widths=[4, 8, 16]) # add switchable precision

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    eos = tokenizer.eos_token if tokenizer is not None else ""

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Shuffle and truncate the train split only
    dataset = raw_datasets["train"].shuffle(seed=42).select(range(MAX_DATASET_SIZE))
    if MAX_DATASET_SIZE > 1:
        # 80/20 split
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split["train"].map(lambda x: sft_preprocess(x, tokenizer))
        eval_dataset = split["test"].map(lambda x: sft_preprocess(x, tokenizer))
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(eval_dataset)}")
    else:
        # Don't split if dataset has only 1 example
        train_dataset = split["train"].map(lambda x: sft_preprocess(x, tokenizer))
        eval_dataset = None
        print(f"Train size: {len(train_dataset)}")
        print("Validation set not created (MAX_DATASET_SIZE <= 1)")
    print("Example preprocessed train sample:")
    print(train_dataset[0]["text"])
    eval_dataset.to_json("/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json") # Save eval set for inference script
    train_dataset.to_json("/content/drive/MyDrive/Colab_Notebooks/eic_llm/train_set.json") # Save train set (for debug)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset, # dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Set bit-widths per layer dynamically (you can randomize or group as needed)
    config1 = {f"transformer.h.{i}": 4 if i % 2 == 0 else 8 for i in range(12)}  # for 12 layers
    config2 = {f"transformer.h.{i}": 4 for i in range(12)}
    config3 = {f"transformer.h.{i}": 8 for i in range(12)}
    if USE_QUANTIZATION:
        # set_active_bitwidths(model, config3) # static training
        set_random_bitwidths(model) # dynamic training

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if USE_BITWISE_LORA:
        torch.save(model.base_model.model.state_dict(), bitwise_lora_adapter_path)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
