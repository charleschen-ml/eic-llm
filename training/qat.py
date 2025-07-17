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
import math

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from transformers import TrainerCallback
from torch.optim import AdamW
from trl import SFTTrainer

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
                 bitwidth_schedule="static",
                 cyclic_repeat_per_bit=1,
                 adapter_path=None):
        self.max_dataset_size = max_dataset_size
        self.use_quantization = use_quantization
        self.use_bitwise_lora = use_bitwise_lora
        self.quant_layers = quant_layers
        self.bit_choices = bit_choices
        self.bitwidth_schedule = bitwidth_schedule  # "static", "cyclic", or "random"
        self.cyclic_repeat_per_bit = cyclic_repeat_per_bit
        self.adapter_path = adapter_path or "/content/drive/MyDrive/Colab_Notebooks/nn/gpt2-qat/full_qat_model.pt"
    
# Settings
r = 16  # LoRA rank
alpha = 32 # LoRA alpha
USE_DEBUG = False

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

def get_static_bitwidth(step, bit_choices):
    # Simply cycle through bitwidths based on step
    bit_index = step % len(bit_choices)
    return bit_choices[bit_index]

def quantize_tensor(tensor, num_bits=4):
    device = tensor.device
    max_val = tensor.detach().abs().max()
    scale = max_val / (2 ** (num_bits - 1) - 1)
    quant = torch.round(tensor / scale).clamp(
        min=-(2 ** (num_bits - 1)), max=2 ** (num_bits - 1) - 1
    )
    dequant = quant * scale
    return dequant.to(device)

# Precompute quantized weights
def patch_linear_forward_with_switchable_quantization(model, bit_widths, quant_layers):
    for name, module in model.named_modules():
        if not any(name.startswith(f"transformer.h.{i}.") for i in quant_layers):
            continue
        if isinstance(module, (nn.Linear, Conv1D)):
            module._quantized_weights = {}  # e.g., {4: tensor, 8: tensor}

            # Precompute quantized weights for bit choices
            for b in bit_widths:
                w = module.weight.detach().clone().to(module.weight.device)
                q_w = quantize_tensor(w, num_bits=b)
                
                # Print quantized stats
                mean_diff = (w - q_w).abs().mean().item()
                max_val = w.max().item()
                min_val = w.min().item()
                if USE_DEBUG:
                    print(
                        f"[Quantize] {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Max abs weight before: {max_val:.4f} | Min abs weight before: {min_val:.4f}")

                # Store precomputed quantized weights
                module._quantized_weights[b] = q_w

            module._active_bit = bit_widths[0]  # set default
            module._bit_choices = bit_widths

# Set default and override bitwidths
def set_active_bitwidths(model, bit_config_dict, default_bit=32):
    print(f"\n[set_active] bit_config_dict = {bit_config_dict} | default_bit = {default_bit}")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
            # Default bit for all layers
            module._active_bit = default_bit

            # Overrides
            for pattern, bit in bit_config_dict.items():
                if pattern in name:
                    module._active_bit = bit
                    break

# Define custom forward, which creates lora at runtime
def add_bitwise_lora_adapters(model, bit_widths, quant_layers):

    for name, module in model.named_modules():
        # Unfreeze LoRA adapter weights listed in QUANT_LAYERS
        if (
            hasattr(module, "_lora_adapters")
            and any(name.startswith(f"transformer.h.{i}.") for i in quant_layers)
        ):
            for adapter in module._lora_adapters.values():
                for submodule in adapter.modules():
                    for param in submodule.parameters(recurse=True):
                        param.requires_grad = True

        # Apply lora to layers that have precomputed quantized weights
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"): # 7/7: removed hasattr(module, "_quantized_weights")
            module._lora_adapters = nn.ModuleDict()
            module._active_bit = bit_widths[0]
            module._bit_choices = bit_widths
            module._layer_name = name
            module._original_forward = module.forward  # Store original forward

            # Custom forward (replaces original forward)
            def forward_with_quant_and_lora(self, input):
                import wandb
                # Use original forward if bit not active
                if getattr(self, "_active_bit", None) is None:
                    return self._original_forward(input)
                
                bit_key = str(self._active_bit) # load bit_key in string format for dict key
                input = input[0] if isinstance(input, tuple) else input # to remove

                # Compute base output (no lora, using base, quantized weights)
                # weight = self._quantized_weights[self._active_bit] # debug: load *static* quantized weights
                # weight = self.weight # debug: load base weights
                weight = quantize_tensor(self.weight, num_bits=self._active_bit) # quantize *backpropped* self.weight
                weight = weight.T # transpose
                bias = self.bias # load bias
                output = F.linear(input, weight, bias) # compute output = input * weight.T + bias

                # Lazy init LoRA adapters at runtime
                if not hasattr(self, "_lora_adapters") or not self._lora_adapters: # if lora doesn't exist yet
                    self._lora_adapters = nn.ModuleDict()
                    in_features = input.shape[-1]
                    out_features = output.shape[-1]
                    for b in self._bit_choices:
                        lora_down = nn.Linear(in_features, r, bias=False).to(input.device)
                        lora_up = nn.Linear(r, out_features, bias=False).to(input.device)
                        nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5)) # 7/7: lora init kick start
                        nn.init.zeros_(lora_up.weight) # 7/7: lora init kick start
                        self._lora_adapters[str(b)] = nn.Sequential(lora_down, lora_up)
                        # print(f"[bitwise_lora] Created lora for layer {self._layer_name} | {b} bits")

                USE_LORA = True # flag for debugging
                if USE_LORA:
                    # Compute lora and add to base output (if adapters exist)
                    if hasattr(self, "_lora_adapters") and bit_key in self._lora_adapters:
                        lora = self._lora_adapters[bit_key]
                        try:
                            lora_out = lora(input)
                            output = output + alpha / r * lora_out # vanilla lora

                            # Monitor LoRA learning for a specific layer in wandb
                            if self._layer_name == "transformer.h.11.mlp.c_fc":
                                lora_up = lora[1]
                                if wandb.run is not None:
                                    wandb.log({
                                        "lora/transformer.h.11.mlp.c_fc/lora_out_norm": lora_out.norm().item(),
                                        "lora/transformer.h.11.mlp.c_fc/lora_weight_norm": lora_up.weight.norm().item(),
                                    })

                        except RuntimeError as e:
                            print(f"[Forward] Skipped {self._layer_name} | Bit: {bit_key} | {e}")
                            # pass
                    else:
                        # print(f"[LoRA] No LoRA adapter for bit {bit_key} in {self._layer_name}")
                        pass

                return output

            # Bind custom forward
            module.forward = forward_with_quant_and_lora.__get__(module, type(module))

# Custom callback to handle bit-width scheduling during training
class BitwidthSchedulingCallback(TrainerCallback):
    def __init__(self, model, bit_choices, bitwidth_schedule, cyclic_repeat_per_bit):
        self.model = model
        self.bit_choices = bit_choices
        self.bitwidth_schedule = bitwidth_schedule  # "static", "cyclic", or "random"
        self.cyclic_repeat_per_bit = cyclic_repeat_per_bit
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_count += 1
        
        if self.bitwidth_schedule == "cyclic":
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
        elif self.bitwidth_schedule == "random": # random, uniform bit assignment
            bit_config = {}
            chosen_bit = random.choice(self.bit_choices) # uniform bit assignment
            for name, module in self.model.named_modules():
                if hasattr(module, "_quantized_weights") and "lm_head" not in name:
                    bit_config[name] = chosen_bit
            set_active_bitwidths(self.model, bit_config)
        else: # "static"
            bit_config = {}
            current_bit = get_static_bitwidth(
                self.step_count, 
                self.bit_choices
            )
            for name, module in self.model.named_modules():
                if hasattr(module, "_quantized_weights") and "lm_head" not in name:
                    bit_config[name] = current_bit
            
            set_active_bitwidths(self.model, bit_config)

# Preprocess (concatenate) squad dataset context, question, answer
def sft_preprocess(example, tokenizer):
    answer = example["answers"]["text"][0].strip()
    return {
        "text": example["context"].strip() + "\n" + example["question"].strip() + "\n" + answer + tokenizer.eos_token
    }

def main(script_args, training_args, model_args, qat_args):
    """
    Main training function with configurable parameters.
    
    Args:
        script_args, training_args, model_args: Standard TRL arguments
        qat_args: QATArguments object containing QAT-specific parameters
    """
    # # Use default QAT arguments if none provided
    # if qat_args is None:
    #     qat_args = QATArguments()

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
        patch_linear_forward_with_switchable_quantization(model, bit_widths = qat_args.bit_choices, quant_layers = qat_args.quant_layers)
        print("After patch:", model.transformer.h[0].mlp.c_fc.forward.__code__)
        add_bitwise_lora_adapters(model, bit_widths = qat_args.bit_choices, quant_layers = qat_args.quant_layers)

    # Set trainable layers
    for param in model.parameters(): # Freeze all layers by default
        param.requires_grad = False
    model.transformer.wte.weight.requires_grad = True # embedding
    model.lm_head.weight.requires_grad = True # language model head
    # for name, param in model.named_parameters(): # layer norm
    #     if "ln_" in name:
    #         param.requires_grad = True
    # model.transformer.wpe.weight.requires_grad = True  # position

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    eos = tokenizer.eos_token if tokenizer is not None else ""
    
    # Debug: Manually set active
    # temp_bit_config_dict = {
    #     f"transformer.h.{i}": 32 for i in range(12)
    # }
    # set_active_bitwidths(model, temp_bit_config_dict)

    # Create lora (lazy init)
    if True: # debug: always create lora
    # if qat_args.use_bitwise_lora:
        # Dummy pass to create lora
        model.eval()
        print("Before patch:", model.transformer.h[11].mlp.c_fc.forward.__code__)
        with torch.no_grad():
            dummy_input = tokenizer("hello world", return_tensors="pt")["input_ids"].to(model.device)
            model(dummy_input)
        print("After patch:", model.transformer.h[11].mlp.c_fc.forward.__code__)

        # Verify lora created
        for name, module in model.named_modules():
            if hasattr(module, "_lora_adapters"):
                print(f"{name}: {list(module._lora_adapters.keys())}")

    # Create Callback
    if qat_args.use_bitwise_lora:
        callbacks = [BitwidthSchedulingCallback(
            model, 
            bit_choices=qat_args.bit_choices,
            bitwidth_schedule=qat_args.bitwidth_schedule,
            cyclic_repeat_per_bit=qat_args.cyclic_repeat_per_bit
        )]
        
        if qat_args.bitwidth_schedule in ["cyclic", "static", "random"]:
            print(f"[Config] Using {qat_args.bitwidth_schedule} bit-width assignment from: {qat_args.bit_choices}")
        else:
            print(f"[Warning] Unknown bitwidth_schedule: {qat_args.bitwidth_schedule}. Defaulting to 'static'")
            qat_args.bitwidth_schedule = "static"
            print(f"[Config] Using {qat_args.bitwidth_schedule} bit-width assignment from: {qat_args.bit_choices}")
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
    eval_dataset.to_json("/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json") # Save eval set for inference script
    train_dataset.to_json("/content/drive/MyDrive/Colab_Notebooks/eic_llm/train_set.json") # Save train set (for debug)

    ################
    # Training
    ################
    trainer = SFTTrainer(
    # trainer=SFTTrainerWithGradLoggingNoWTE( # 7/6: allow print grad norm AND prevent wte grad updates
        model=model,
        args=training_args,
        train_dataset=train_dataset, # dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset, # dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        # peft_config=get_peft_config(model_args), # shouldn't be used but removing just to be sure
        callbacks = callbacks
    )

    # Print trainable parameters before training
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"[Trainable] {name}, shape: {param.shape}")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"[Frozen] {name}, shape: {param.shape}")

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

    # Train
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
    parser.add_argument("--bitwidth_schedule", type=str, default="static", choices=["static", "cyclic", "random"],
                       help="Bit-width scheduling strategy: 'static' (default), 'cyclic', or 'random'")
    parser.add_argument("--cyclic_repeat_per_bit", type=int, default=1,
                       help="Number of times to repeat each bit-width (only used with cyclic schedule)")
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
        quant_layers=args.quant_layers,
        bit_choices=args.bit_choices,
        bitwidth_schedule=args.bitwidth_schedule,
        cyclic_repeat_per_bit=args.cyclic_repeat_per_bit,
        adapter_path=args.adapter_path
    )
    
    main(script_args, training_args, model_args, qat_args)
