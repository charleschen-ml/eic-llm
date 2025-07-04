# Copied from TRL to be run from Colab

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
import torch.nn.functional as F
import re
import os
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # To fix torch deterministic error
torch.use_deterministic_algorithms(True)
import random

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from transformers import TrainerCallback

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

# Custom arguments for QAT-specific parameters
class QATArguments:
    def __init__(self, 
                 max_dataset_size=2,
                 use_quantization=True,
                 use_bitwise_lora=True,
                 quant_layers=[6, 8, 10, 11],
                 bit_choices=[8, 16],
                 use_cyclic_bitwidth=True,
                 cyclic_repeat_per_bit=1,
                 adapter_path=None):
        self.max_dataset_size = max_dataset_size
        self.use_quantization = use_quantization
        self.use_bitwise_lora = use_bitwise_lora
        self.quant_layers = quant_layers
        self.bit_choices = bit_choices
        self.use_cyclic_bitwidth = use_cyclic_bitwidth
        self.cyclic_repeat_per_bit = cyclic_repeat_per_bit
        self.adapter_path = adapter_path or "/content/drive/MyDrive/Colab_Notebooks/nn/gpt2-qat/full_qat_model.pt"
    

def get_cyclic_bitwidth(step, bit_choices, repeat_per_bit=1):
    """
    Compute cyclic bit-width based on current training step.
    Loops through bit choices from min to max and back, with configurable repeats per bit.
    
    Args:
        step: Current training step
        bit_choices: List of valid bit-width choices
        repeat_per_bit: Number of times to repeat each bit-width before moving to next
    
    Returns:
        Current bit-width for the cycle (integer from bit_choices)
    """
    # Calculate total steps for one complete cycle

    total_cycle_steps = (2 * len(bit_choices) - 2) * repeat_per_bit
    forward_steps = len(bit_choices) * repeat_per_bit
        # Calculate position within the cycle
    cycle_step = step % total_cycle_steps
    
    if cycle_step < forward_steps:
        # Forward phase: min to max
        bit_index = cycle_step // repeat_per_bit
    else:
        # Backward phase: max to min
        backward_step = cycle_step - forward_steps
        bit_index = len(bit_choices) - 2 - backward_step // repeat_per_bit
    
    return bit_choices[bit_index]

def quantize_tensor(tensor, num_bits=4) -> object:
    device = tensor.device # capture tensor device (gpu)
    max_val = tensor.abs().max()
    scale = max_val / (2 ** (num_bits - 1) - 1)
    tensor_quant = torch.round(tensor / scale).clamp(
        min=-(2 ** (num_bits - 1)), max=2 ** (num_bits - 1) - 1
    )
    tensor_dequant = tensor_quant * scale
    return tensor_dequant.to(device) # move tensor to gpu

def patch_linear_forward_with_switchable_quantization(model, bit_widths, quant_layers):
    """
    For each nn.Linear layer, store quantized weights for multiple bit-widths
    and use a runtime flag to choose the active one.
    """
    print(f"[Quantize] Patching linear forward with switchable quantization for layers: {quant_layers}")
    for name, module in model.named_modules():
        if not any(name.startswith(f"transformer.h.{i}.") for i in quant_layers):
            continue
        if any(skip in name for skip in ["lm_head", "wte"]):
            continue  # ✅ skip output and embedding layers
        if isinstance(module, (nn.Linear, Conv1D)):
            module._quantized_weights = {}  # e.g., {4: tensor, 8: tensor}

            # Precompute quantized weights for bit choices
            for b in bit_widths:
                w = module.weight.detach().clone().to(module.weight.device)
                q_w = quantize_tensor(w, num_bits=b)
                mean_diff = (w - q_w).abs().mean().item()
                max_before = w.abs().max().item()
                print(
                    f"[Quantize] Precomputed {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Max abs weight before: {max_before:.4f}")
                module._quantized_weights[b] = q_w

            module._active_bit = bit_widths[0]  # default
            module._bit_choices = bit_widths

            def quantized_forward(self, input):
                weight = self._quantized_weights[self._active_bit]
                if weight.shape[1] != input.shape[-1]: 
                    weight = weight.T # transpose if dim mismatch (for gpt2 internal layers e.g. c_attn)
                # print(f"[Forward] {self} | Bit: {self._active_bit} | Weight shape: {weight.shape}")
                return nn.functional.linear(input, weight, # put bias on the same device as the layer itself
                                            self.bias.to(input.device) if self.bias is not None else None)

            module.forward = quantized_forward.__get__(module, nn.Linear)

# def set_active_bitwidths(model, bit_config_dict): # To remove
#     print(f"\n[set_active] start: {bit_config_dict}")  # debug
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
#             # print(f"[set_active] {name} is linear and has q_weights")
#             if name in bit_config_dict:
#                 module._active_bit = bit_config_dict[name]
#                 print(f"[set_active] successfully configured: {name} | Active bit: {bit_config_dict[name]}")

def set_active_bitwidths(model, bit_config_dict):
    print(f"\n[set_active] start: {bit_config_dict}")  # debug
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
            # Always skip deactivating c_attn layers
            if "c_attn" in name:
                continue  # ✅ always leave c_attn active

            # Default all layers to inactive
            module._active_bit = None

            # Only activate layers that are explicitly configured
            for prefix, bit in bit_config_dict.items():
                if name.startswith(prefix):
                    module._active_bit = bit  # ✅ only activate if explicitly configured
                    # print(f"[set_active] set {name} to {bit} bits")

# def add_bitwise_lora_adapters(model, bit_widths=[4, 8, 16]):
#     """
#     For each Linear layer in transformer.h.0, attach multiple LoRA adapters — one per bit-width.
#     During forward pass, apply quantized weight and the matching LoRA adapter.
#     """
#     for name, module in model.named_modules():
#         # Only apply each linear layer in this module
#         if not any(name.startswith(f"transformer.h.{i}.") for i in [11]):
#             continue

#         # Apply only to Linear layers that were quantized
#         if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
#             module._lora_adapters = nn.ModuleDict()
#             # Create one LoRA module per bit-width (e.g., 4-bit and 8-bit)
#             for b in bit_widths:
#                 print(f"[bitwise_lora] {name} | shape = {module.weight.shape}") # debug
#                 r = 8  # LoRA rank; can tune this
#                 if module.weight.shape[1] == module.weight.shape[0] * 3:
#                     print(f"[LoRA WARNING] Skipping {name} due to shape mismatch risk.")
#                     continue
#                 in_features = module.weight.shape[1]
#                 out_features = module.weight.shape[0]
#                 lora_down = nn.Linear(in_features, r, bias=False)
#                 lora_up = nn.Linear(r, out_features, bias=False)
#                 print(f"[bitwise_lora] lora_down shape: {lora_down.weight.shape}") # debug
#                 print(f"[bitwise_lora] lora_up shape: {lora_up.weight.shape}") # debug
#                 module._lora_adapters[str(b)] = nn.Sequential(lora_down, lora_up)
#                 print(f"[bitwise_lora] Created lora for layer {name} | {b} bits")

#             # Set default active bit-width
#             module._active_bit = bit_widths[0]

#             module._layer_name = name # temp debug

#             def forward_with_quant_and_lora(module, input):
#                 """
#                 Custom forward function that applies quantized weights and the corresponding LoRA adapter.
#                 Assumes input is a tuple, as passed into nn.Module.forward hooks.
#                 """
#                 bit_key = str(module._active_bit)
#                 print(f"[Forward] {module._layer_name} | Bit: {bit_key}")
#                 input = input[0] if isinstance(input, tuple) else input
#                 print(f"[Forward] input shape: {input.shape}") # debug

#                 weight = module._quantized_weights[module._active_bit]
#                 print(f"[Forward] weight shape: {weight.shape}") # debug

#                 # Transpose if needed for compatibility
#                 if weight.shape[1] != input.shape[-1]:
#                     weight = weight.T
#                     print(f"[Forward] transposed weight shape: {weight.shape}") # debug
#                 bias = module.bias

#                 # Quantized output
#                 output = F.linear(input, weight, bias)
#                 print(f"[Forward] output shape: {output.shape}") # debug

#                 # Apply LoRA if available and compatible
#                 if hasattr(module, "_lora_adapters") and module._lora_adapters:

#                     lora = module._lora_adapters[bit_key] if bit_key in module._lora_adapters else None
#                     print(f"[Forward] {module._layer_name} | Bit: {bit_key} | lora attr exists")
#                     if lora is not None:
#                         print(f"[Forward] {module._layer_name} | Bit: {bit_key} | lora exists")
#                         try:
#                             lora_out = lora(input)
#                             output += lora_out
#                             print(f"[Forward] computed {module._layer_name} | Bit: {bit_key}")
#                         except RuntimeError as e:
#                             print(f"[Forward] skipped {module._layer_name} | Bit: {bit_key} | {e}")
#                     else:
#                         print(f"[LoRA] No LoRA adapter for bit {module._active_bit} in {module}")
#                 return output

#             # Replace original forward function
#             module.forward = forward_with_quant_and_lora.__get__(module, type(module)) # type(module) to include conv1D

def add_bitwise_lora_adapters(model, bit_widths, quant_layers):
    """
    For each Linear layer in transformer.h.0, attach multiple LoRA adapters — one per bit-width.
    During forward pass, apply quantized weight and the matching LoRA adapter.
    """
    for name, module in model.named_modules():
        # Only apply each linear layer in this module
        if not any(name.startswith(f"transformer.h.{i}.") for i in quant_layers):
            continue

        # Apply only to Linear layers that were quantized
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
            module._lora_adapters = nn.ModuleDict()
            module._active_bit = bit_widths[0]
            module._bit_choices = bit_widths
            module._layer_name = name  # temp debug

            def forward_with_quant_and_lora(self, input):
                if getattr(self, "_active_bit", None) is None:
                    return input
                bit_key = str(self._active_bit)
                # print(f"[Forward] {self._layer_name} | Bit: {bit_key}")
                input = input[0] if isinstance(input, tuple) else input
                # print(f"[Forward] input shape: {input.shape}")  # debug

                weight = self._quantized_weights[self._active_bit]
                # print(f"[Forward] weight shape: {weight.shape}")  # debug

                if weight.shape[1] != input.shape[-1]:
                    weight = weight.T
                    # print(f"[Forward] transposed weight shape: {weight.shape}")  # debug

                bias = self.bias
                output = F.linear(input, weight, bias)
                # print(f"[Forward] output shape: {output.shape}")  # debug

                # Lazy init LoRA adapters
                if not hasattr(self, "_lora_adapters") or not self._lora_adapters:
                    self._lora_adapters = nn.ModuleDict()
                    r = 8  # LoRA rank; can tune this
                    in_features = input.shape[-1]
                    out_features = output.shape[-1]
                    for b in self._bit_choices:
                        # print(f"[bitwise_lora] {self._layer_name} | shape = {self.weight.shape}")  # debug
                        if self.weight.shape[1] == self.weight.shape[0] * 3:
                            # print(f"[bitwise_lora] Skipping {self._layer_name} due to shape mismatch risk.")
                            continue
                        lora_down = nn.Linear(in_features, r, bias=False).to(input.device)
                        lora_up = nn.Linear(r, out_features, bias=False).to(input.device)
                        # print(f"[bitwise_lora] lora_down shape: {lora_down.weight.shape}")  # debug
                        # print(f"[bitwise_lora] lora_up shape: {lora_up.weight.shape}")  # debug
                        self._lora_adapters[str(b)] = nn.Sequential(lora_down, lora_up)
                        # print(f"[bitwise_lora] Created lora for layer {self._layer_name} | {b} bits")

                # Apply LoRA if available and compatible
                if hasattr(self, "_lora_adapters") and bit_key in self._lora_adapters:
                    # print(f"[Forward] {self._layer_name} | Bit: {bit_key} | lora attr exists")
                    lora = self._lora_adapters[bit_key]
                    # print(f"[Forward] {self._layer_name} | Bit: {bit_key} | lora exists")
                    try:
                        lora_out = self._lora_adapters[bit_key](input)
                        output += lora_out
                        # print(f"[Forward] Computed {self._layer_name} | Bit: {bit_key}")
                    except RuntimeError as e:
                        # print(f"[Forward] Skipped {self._layer_name} | Bit: {bit_key} | {e}")
                        pass
                else:
                    # print(f"[LoRA] No LoRA adapter for bit {bit_key} in {self._layer_name}")
                    pass

                return output

            module.forward = forward_with_quant_and_lora.__get__(module, type(module))

# Custom callback to handle bit-width scheduling during training
class BitwidthSchedulingCallback(TrainerCallback):
    def __init__(self, model, bit_choices, use_cyclic, cyclic_repeat_per_bit):
        self.model = model
        self.bit_choices = bit_choices
        self.use_cyclic = use_cyclic 
        self.cyclic_repeat_per_bit = cyclic_repeat_per_bit
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_count += 1
        
        if self.use_cyclic:
            # Use cyclic scheduling with automatic period based on bit choices
            current_bit = get_cyclic_bitwidth(
                self.step_count, 
                self.bit_choices,
                self.cyclic_repeat_per_bit
            )
            
            # Apply the same bit-width to all quantized layers
            bit_config = {}
            for name, module in self.model.named_modules():
                if hasattr(module, "_quantized_weights") and "lm_head" not in name:
                    bit_config[name] = current_bit
            
            if self.step_count % 10 == 0:  # Log every 10 steps to avoid spam
                print(f"[CyclicBitwidth] Step {self.step_count}: Using {current_bit}-bit quantization")
            
            set_active_bitwidths(self.model, bit_config)
        else:
            # Use random bit-width assignment (original behavior)
            bit_config = {}
            for name, module in self.model.named_modules():
                if hasattr(module, "_quantized_weights") and "lm_head" not in name:
                    chosen_bit = random.choice(self.bit_choices)
                    bit_config[name] = chosen_bit
            set_active_bitwidths(self.model, bit_config)

# # Custom callback to randomize bitwidths before each train step (original random implementation)
# class BitwidthRandomizationCallback(TrainerCallback):
#     def __init__(self, model, bit_choices=BIT_CHOICES):
#         self.model = model
#         self.bit_choices = bit_choices

#     def on_step_begin(self, args, state, control, **kwargs):
#         # Randomly assign a bitwidth to each supported layer
#         bit_config = {}
#         for name, module in self.model.named_modules():
#             if hasattr(module, "_quantized_weights") and "lm_head" not in name:
#                 chosen_bit = random.choice(self.bit_choices)
#                 # print(f"[BitwidthRandomization] Prepare {name} <- {chosen_bit} bit")
#                 bit_config[name] = chosen_bit
#         set_active_bitwidths(self.model, bit_config)

def sft_preprocess(example, tokenizer):
    answer = example["answers"]["text"][0].strip()
    return {
        "text": example["context"].strip() + "\n" + example["question"].strip() + "\n" + answer + tokenizer.eos_token
    }

def main(script_args, training_args, model_args, qat_args=None):
    """
    Main training function with configurable parameters.
    
    Args:
        script_args, training_args, model_args: Standard TRL arguments
        qat_args: QATArguments object containing QAT-specific parameters
    """
    # Use default QAT arguments if none provided
    if qat_args is None:
        qat_args = QATArguments()

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
    if qat_args.use_quantization:
        model.to("cuda")  # ✅ move to GPU before quantizing
        print("Before patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        patch_linear_forward_with_switchable_quantization(model, qat_args.bit_choices, qat_args.quant_layers)
        print("After patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        add_bitwise_lora_adapters(model, qat_args.bit_choices, qat_args.quant_layers)
    

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    eos = tokenizer.eos_token if tokenizer is not None else ""
    
    # Dummy forward to create LoRA modules
    if qat_args.use_bitwise_lora:
        callbacks = [BitwidthSchedulingCallback(
            model, 
            bit_choices=qat_args.bit_choices,
            use_cyclic=qat_args.use_cyclic_bitwidth,
            cyclic_repeat_per_bit=qat_args.cyclic_repeat_per_bit
        )]
        
        if qat_args.use_cyclic_bitwidth:
            print(f"[Config] Using cyclic bit-width scheduling: {min(qat_args.bit_choices)}-{max(qat_args.bit_choices)} bits")
        else:
            print(f"[Config] Using random bit-width assignment from: {qat_args.bit_choices}")
        
        # Dummy pass to create lora
        model.eval()
        with torch.no_grad():
            dummy_input = tokenizer("hello world", return_tensors="pt")["input_ids"].to(model.device)
            model(dummy_input)
        
        # Verify lora created
        for name, module in model.named_modules(): 
            if hasattr(module, "_lora_adapters"):
                print(f"{name}: {list(module._lora_adapters.keys())}")
    else:
        callbacks = []        

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Shuffle and truncate the train split only
    dataset = raw_datasets["train"].shuffle(seed=42).select(range(qat_args.max_dataset_size))
    if qat_args.max_dataset_size > 1:
        # 80/20 split
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split["train"].map(lambda x: sft_preprocess(x, tokenizer))
        eval_dataset = split["test"].map(lambda x: sft_preprocess(x, tokenizer))
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation size: {len(eval_dataset)}")
    else:
        # Don't split if dataset has only 1 example
        train_dataset = dataset.map(lambda x: sft_preprocess(x, tokenizer))
        eval_dataset = None
        print(f"Train size: {len(train_dataset)}")
        print(f"Validation set not created (max_dataset_size <= 1)")
    print("Example preprocessed train sample:")
    print(train_dataset[0]["text"])
    eval_dataset.to_json("/content/drive/MyDrive/Colab_Notebooks/nn/eic_llm/eval_set.json") # Save eval set for inference script
    train_dataset.to_json("/content/drive/MyDrive/Colab_Notebooks/nn/eic_llm/train_set.json") # Save train set (for debug)

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
        callbacks = callbacks
    )

    # # Set bit-widths per layer dynamically (you can randomize or group as needed)
    # config1 = {f"transformer.h.{i}": 4 if i % 2 == 0 else 8 for i in range(12)}  # for 12 layers
    # config2 = {f"transformer.h.{i}": 4 for i in range(12)}
    # config3 = {f"transformer.h.{i}": 8 for i in range(12)}
    # config4 = {f"transformer.h.11": 8}
    # if qat_args.use_quantization and not qat_args.use_bitwise_lora:
    #     set_active_bitwidths(model, config4) # static training
    #     # set_random_bitwidths(model) # dynamic training (deprecated)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if qat_args.use_bitwise_lora:
        print(type(model))                         # should show PeftModel
        print(type(model.base_model))             # should show PeftModelForCausalLM
        # print(type(model.base_model.model))       # should show GPT2LMHeadModel
        torch.save(model.state_dict(), qat_args.adapter_path)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    
    # Add QAT-specific arguments
    parser.add_argument("--max_dataset_size", type=int, default=2, 
                       help="Total samples (train+validation). Set to >= 2.")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                       help="Whether to apply quantization")
    parser.add_argument("--no_quantization", dest="use_quantization", action="store_false",
                       help="Disable quantization")
    parser.add_argument("--use_bitwise_lora", action="store_true", default=True,
                       help="Whether to use bitwise LoRA adapters")
    parser.add_argument("--no_bitwise_lora", dest="use_bitwise_lora", action="store_false",
                       help="Disable bitwise LoRA adapters")
    parser.add_argument("--quant_layers", type=str, default="6,8,10,11",
                       help="Comma-separated list of h.* layers to quantize")
    parser.add_argument("--bit_choices", type=str, default="8,16",
                       help="Comma-separated list of bit choices for LoRA")
    parser.add_argument("--use_cyclic_bitwidth", action="store_true", default=True,
                       help="Enable cyclic bit-width scheduling")
    parser.add_argument("--no_cyclic_bitwidth", dest="use_cyclic_bitwidth", action="store_false",
                       help="Disable cyclic bit-width scheduling")
    parser.add_argument("--cyclic_repeat_per_bit", type=int, default=1,
                       help="Number of times to repeat each bit-width")
    parser.add_argument("--adapter_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/nn/gpt2-qat/full_qat_model.pt",
                       help="Path to save the bitwise LoRA adapter")
    
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Parse QAT-specific arguments
    args = parser.parse_args()
    
    # Convert string arguments to lists
    quant_layers = [int(x.strip()) for x in args.quant_layers.split(",")]
    bit_choices = [int(x.strip()) for x in args.bit_choices.split(",")]
    
    # Create QAT arguments object
    qat_args = QATArguments(
        max_dataset_size=args.max_dataset_size,
        use_quantization=args.use_quantization,
        use_bitwise_lora=args.use_bitwise_lora,
        quant_layers=quant_layers,
        bit_choices=bit_choices,
        use_cyclic_bitwidth=args.use_cyclic_bitwidth,
        cyclic_repeat_per_bit=args.cyclic_repeat_per_bit,
        adapter_path=args.adapter_path
    )
    
    main(script_args, training_args, model_args, qat_args)
