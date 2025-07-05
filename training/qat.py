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

# Settings
MAX_DATASET_SIZE = 2  # Total samples (train+validation). Set to >= 2.
USE_QUANTIZATION = True
USE_BITWISE_LORA = True
QUANT_LAYERS = [11] # h.* layers to quantize
BIT_CHOICES = [2] # bit choices for LoRA

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

def patch_linear_forward_with_switchable_quantization(model, bit_widths=BIT_CHOICES):
    """
    For each nn.Linear layer, store quantized weights for multiple bit-widths
    and use a runtime flag to choose the active one.
    """
    for name, module in model.named_modules():
        if not any(name.startswith(f"transformer.h.{i}.") for i in QUANT_LAYERS):
            continue
        if any(skip in name for skip in ["lm_head", "wte"]):
            continue  # ✅ skip output and embedding layers
        if isinstance(module, (nn.Linear, Conv1D)):
            module._quantized_weights = {}  # e.g., {4: tensor, 8: tensor}

            # Precompute quantized weights
            for b in bit_widths:
                w = module.weight.detach().clone().to(module.weight.device)
                q_w = quantize_tensor(w, num_bits=b)
                w_mean = w.mean().item()
                q_w_mean = q_w.mean().item()
                mean_diff = (w - q_w).abs().mean().item()
                max_val = w.max().item()
                min_val = w.min().item()
                print(
                    f"[Quantize] Precomputed {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Max abs weight before: {max_val:.4f} | Min abs weight before: {min_val:.4f} | Mean weight before: {w_mean:.4f} | Mean quantized weight: {q_w_mean:.4f}")
                
                ###
                # Flatten tensors
                w_flat = w.flatten()
                q_w_flat = q_w.flatten()

                # Masked means
                pos_w_mean = w_flat[w_flat > 0].mean().item() if (w_flat > 0).any() else 0.0
                neg_w_mean = w_flat[w_flat < 0].mean().item() if (w_flat < 0).any() else 0.0
                pos_qw_mean = q_w_flat[q_w_flat > 0].mean().item() if (q_w_flat > 0).any() else 0.0
                neg_qw_mean = q_w_flat[q_w_flat < 0].mean().item() if (q_w_flat < 0).any() else 0.0

                print(f"[Quantize] {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Min: {min_val:.6f} | Max: {max_val:.6f}")
                print(f"[Quantize] {name} | Mean weight before: {w_mean:.4f} | Mean quantized weight: {q_w_mean:.4f}")
                print(f"[Quantize] {name} | Avg pos before: {pos_w_mean:.4f} | Avg neg before: {neg_w_mean:.4f}")
                print(f"[Quantize] {name} | Avg pos quant:  {pos_qw_mean:.4f} | Avg neg quant:  {neg_qw_mean:.4f}")
                print(f"[Quantize] {name} | First 10 elements (original):  {w_flat[:10].tolist()}")
                print(f"[Quantize] {name} | First 10 elements (quantized): {q_w_flat[:10].tolist()}")
                ###
                
                module._quantized_weights[b] = q_w

            module._active_bit = bit_widths[0]  # default
            module._bit_choices = bit_widths

            # To remove: now replaced with forward_with_quant_and_lora
            # def quantized_forward(self, input):
            #     weight = self._quantized_weights[self._active_bit]
            #     if weight.shape[1] != input.shape[-1]: 
            #         weight = weight.T # transpose if dim mismatch (for gpt2 internal layers e.g. c_attn)
            #     # print(f"[Forward] {self} | Bit: {self._active_bit} | Weight shape: {weight.shape}")
            #     return nn.functional.linear(input, weight, # put bias on the same device as the layer itself
            #                                 self.bias.to(input.device) if self.bias is not None else None)

            # module.forward = quantized_forward.__get__(module, nn.Linear)

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

def add_bitwise_lora_adapters(model, bit_widths=BIT_CHOICES):
    """
    For each Linear layer in transformer.h.0, attach multiple LoRA adapters — one per bit-width.
    During forward pass, apply quantized weight and the matching LoRA adapter.
    """
    for name, module in model.named_modules():
        # 7/5: freeze everything
        # for param in module.parameters(recurse=True):
        #     param.requires_grad = False

        # Only apply each linear layer in this module
        if not any(name.startswith(f"transformer.h.{i}.") for i in QUANT_LAYERS):
            for param in module.parameters(recurse=True):
                param.requires_grad = False
            continue

        # 7/5 then enable specified ones
        for param in module.parameters(recurse=True):
            param.requires_grad = True

        # Apply only to Linear layers that were quantized
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
            # ⬇️ Insert this to check if bias is still trainable
            # if hasattr(module, "bias") and module.bias is not None:
            #     module.bias.requires_grad = False
            #     print(f"[Freeze] Bias frozen for {name}")
            module._lora_adapters = nn.ModuleDict()
            module._active_bit = bit_widths[0]
            module._bit_choices = bit_widths
            module._layer_name = name  # temp debug

            def forward_with_quant_and_lora(self, input):
                # If _active_bit is not set, use original forward
                if getattr(self, "_active_bit", None) is None:
                    return self._original_forward(input)
                
                bit_key = str(self._active_bit)
                # print(f"[Forward] {self._layer_name} | Bit: {bit_key}")
                input = input[0] if isinstance(input, tuple) else input
                # print(f"[Forward] input shape: {input.shape}")  # debug

                weight = self._quantized_weights[self._active_bit]
                # weight.data.fill_(123.456) # intentionally break it
                # print(f"[Hello] {self._layer_name} | Bit: {self._active_bit} | First 5 weights: {weight.view(-1)[:5]}")
                # print(
                #     f"[Hello] Unique: {torch.unique(weight).numel()} | Min: {weight.min():.2f} | Max: {weight.max():.2f}")
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
                        print(f"[Forward] Computed {self._layer_name} | Bit: {bit_key}")
                    except RuntimeError as e:
                        print(f"[Forward] Skipped {self._layer_name} | Bit: {bit_key} | {e}")
                        # pass
                else:
                    # print(f"[LoRA] No LoRA adapter for bit {bit_key} in {self._layer_name}")
                    pass

                return output
            module._original_forward = module.forward # Store original forward
            module.forward = forward_with_quant_and_lora.__get__(module, type(module))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Still Trainable] {name}, shape: {param.shape}")

# def set_random_bitwidths(model, bit_choices=[4, 8]): # To remove
#     for name, module in model.named_modules():
#         if name.startswith("transformer.h.0") and hasattr(module, "_quantized_weights"):
#             module._active_bit = random.choice(bit_choices)

# Custom callback to randomize bitwidths before each train step
class BitwidthRandomizationCallback(TrainerCallback):
    def __init__(self, model, bit_choices=BIT_CHOICES):
        self.model = model
        self.bit_choices = bit_choices

    def on_step_begin(self, args, state, control, **kwargs):
        # Randomly assign a bitwidth to each supported layer
        bit_config = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "_quantized_weights") and "lm_head" not in name:
                chosen_bit = random.choice(self.bit_choices)
                # print(f"[BitwidthRandomization] Prepare {name} <- {chosen_bit} bit")
                bit_config[name] = chosen_bit
        set_active_bitwidths(self.model, bit_config)

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
        print("Before patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        patch_linear_forward_with_switchable_quantization(model, bit_widths=BIT_CHOICES)
        print("After patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        add_bitwise_lora_adapters(model, bit_widths=BIT_CHOICES) # add switchable precision
    

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    eos = tokenizer.eos_token if tokenizer is not None else ""
    
    # Dummy forward to create LoRA modules
    if USE_BITWISE_LORA:
        # Use callback to randomize bitwidths before each train step
        callbacks = [BitwidthRandomizationCallback(model, bit_choices=BIT_CHOICES)]
        
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
        callbacks = callbacks
    )

    # # Set bit-widths per layer dynamically (you can randomize or group as needed)
    # config1 = {f"transformer.h.{i}": 4 if i % 2 == 0 else 8 for i in range(12)}  # for 12 layers
    # config2 = {f"transformer.h.{i}": 4 for i in range(12)}
    # config3 = {f"transformer.h.{i}": 8 for i in range(12)}
    # config4 = {f"transformer.h.11": 8}
    # if USE_QUANTIZATION and not USE_BITWISE_LORA:
    #     set_active_bitwidths(model, config4) # static training
    #     # set_random_bitwidths(model) # dynamic training (deprecated)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Trainable] {name}, shape: {param.shape}")

    # Print created lora
    print("\n🔍 Created LoRA Adapters:")
    for name, module in model.named_modules():
        if hasattr(module, "_lora_adapters"):
            for bw, lora in module._lora_adapters.items():
                weights = list(lora.parameters())
                norm = sum(p.norm().item() for p in weights)
                print(f"{name} | {bw}-bit LoRA norm: {norm:.4f}")

    # Print active lora
    print("\n🔍 Active LoRA Adapters:")
    for name, module in model.named_modules():
        if hasattr(module, "_lora_adapters") and hasattr(module, "_active_bit"):
            print(
                f"{name} | Active bitwidth: {module._active_bit} | Available: {list(module._lora_adapters.keys())}")

    # Debug 7/5: Check if output requires grad
    inputs = tokenizer("dummy text", return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    if isinstance(outputs, dict) and "logits" in outputs:
        output_tensor = outputs["logits"]
    else:
        output_tensor = outputs[0]  # fallback

    print("Output requires_grad:", output_tensor.requires_grad)

    # Debug 7/5: see if loss function requires grad
    import types
    original_training_step = trainer.training_step
    def wrapped_training_step(self, model, inputs, num_steps):
        outputs = model(**inputs)
        # Try all ways the loss might be computed
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            raise ValueError("Cannot extract loss from model outputs")
        print("🚨 Loss type:", type(loss))
        print("🚨 Loss requires_grad:", loss.requires_grad)
        print("🚨 Loss grad_fn:", loss.grad_fn)
        print("🚨 Any trainable param?", any(p.requires_grad for p in model.parameters()))
        # Continue normally
        self.accelerator.backward(loss)
        return loss.detach()  # what HF Trainer expects
    trainer.training_step = types.MethodType(wrapped_training_step, trainer)

    # Train
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if USE_BITWISE_LORA:
        print(type(model))                         # should show PeftModel
        print(type(model.base_model))             # should show PeftModelForCausalLM
        # print(type(model.base_model.model))       # should show GPT2LMHeadModel
        torch.save(model.state_dict(), bitwise_lora_adapter_path)
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
