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

# Symmetric min-max quantization
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
                w_mean = w.mean().item()
                q_w_mean = q_w.mean().item()
                mean_diff = (w - q_w).abs().mean().item()
                max_val = w.max().item()
                min_val = w.min().item()
                w_flat = w.flatten()
                q_w_flat = q_w.flatten()
                pos_w_mean = w_flat[w_flat > 0].mean().item() if (w_flat > 0).any() else 0.0
                neg_w_mean = w_flat[w_flat < 0].mean().item() if (w_flat < 0).any() else 0.0
                pos_qw_mean = q_w_flat[q_w_flat > 0].mean().item() if (q_w_flat > 0).any() else 0.0
                neg_qw_mean = q_w_flat[q_w_flat < 0].mean().item() if (q_w_flat < 0).any() else 0.0
                print(
                    f"[Quantize] {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Max abs weight before: {max_val:.4f} | Min abs weight before: {min_val:.4f}")
                # print(f"[Quantize] {name} | Bits: {b} | Mean abs diff: {mean_diff:.6f} | Min: {min_val:.6f} | Max: {max_val:.6f}")
                # print(f"[Quantize] {name} | Mean weight before: {w_mean:.4f} | Mean quantized weight: {q_w_mean:.4f}")
                # print(f"[Quantize] {name} | Avg pos before: {pos_w_mean:.4f} | Avg neg before: {neg_w_mean:.4f}")
                # print(f"[Quantize] {name} | Avg pos quant:  {pos_qw_mean:.4f} | Avg neg quant:  {neg_qw_mean:.4f}")
                # print(f"[Quantize] {name} | First 10 elements (original):  {w_flat[:10].tolist()}")
                # print(f"[Quantize] {name} | First 10 elements (quantized): {q_w_flat[:10].tolist()}")

                # Store precomputed quantized weights
                module._quantized_weights[b] = q_w

            module._active_bit = bit_widths[0]  # set default
            module._bit_choices = bit_widths

def set_active_bitwidths(model, bit_config_dict):
    print(f"\n[set_active] start: {bit_config_dict}")  # debug
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)) and hasattr(module, "_quantized_weights"):
            # Skip c_attn layers
            if "c_attn" in name:
                continue

            # Default all layers to inactive
            module._active_bit = None

            # Only activate layers that are explicitly configured
            for prefix, bit in bit_config_dict.items():
                if name.startswith(prefix):
                    module._active_bit = bit  # ✅ only activate if explicitly configured
                    # print(f"[set_active] set {name} to {bit} bits")

# 1. Configures requires_grad for all layers
# 2. Define custom forward, which creates lora at runtime
def add_bitwise_lora_adapters(model, bit_widths, quant_layers):
    # Freeze all layers by default
    for param in model.parameters():
        param.requires_grad = False

    # Workaround: Set requires_grad = True for WTE layer required for gradient flow
    # Weight updates to this layer is later disabled to train only lora layers
    model.transformer.wte.weight.requires_grad = True  # required to avoid loss.requires_grad = False

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
                # Use original forward if bit not active
                if getattr(self, "_active_bit", None) is None:
                    return self._original_forward(input)
                
                bit_key = str(self._active_bit) # load bit_key in string format for dict key
                input = input[0] if isinstance(input, tuple) else input # to remove

                # Compute base output (no lora, using base, quantized weights)
                weight = self._quantized_weights[self._active_bit] # load quantized weights
                # weight = self.weight # load base weights
                if weight.shape[1] != input.shape[-1]: # transpose matrix for special layers
                    weight = weight.T
                bias = self.bias # load bias
                output = F.linear(input, weight, bias) # compute output = input * weight.T + bias

                # Print base L2 norms
                print(f"[{self._layer_name}] base: (input @ weight.T + bias)")
                # print(f"  input shape: {input.shape}")
                # print(f"  weight.T shape: {weight.T.shape}")
                # print(f"  bias shape: {bias.shape if bias is not None else 'None'}")
                print(f"  base_out norm: {output.norm().item():.6f} | mean: {output.mean().item():.6f}")

                # Lazy init LoRA adapters at runtime
                if not hasattr(self, "_lora_adapters") or not self._lora_adapters: # if lora doesn't exist yet
                    self._lora_adapters = nn.ModuleDict()
                    r = 128  # LoRA rank; can tune this
                    in_features = input.shape[-1]
                    out_features = output.shape[-1]
                    for b in self._bit_choices:
                        if self.weight.shape[1] == self.weight.shape[0] * 3: # skip c_attn
                            # print(f"[bitwise_lora] Skipping {self._layer_name} due to shape mismatch risk.")
                            continue
                        lora_down = nn.Linear(in_features, r, bias=False).to(input.device)
                        lora_up = nn.Linear(r, out_features, bias=False).to(input.device)
                        nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5)) # 7/7: lora init kick start
                        nn.init.zeros_(lora_up.weight) # 7/7: lora init kick start
                        self._lora_adapters[str(b)] = nn.Sequential(lora_down, lora_up)
                        # print(f"[bitwise_lora] Created lora for layer {self._layer_name} | {b} bits")

                # Compute lora and add to base output (if adapters exist)
                if hasattr(self, "_lora_adapters") and bit_key in self._lora_adapters:
                    lora = self._lora_adapters[bit_key]
                    try:
                        # lora_out = lora(input)
                        # output += lora_out

                        # Print base+lora L2 norms
                        lora_down = lora[0]
                        lora_up = lora[1]
                        z = lora_down(input)
                        lora_out = lora_up(z)
                        print(f"  output (base) norm: {output.norm().item():.6f} | mean: {output.mean().item():.6f}")
                        output += lora_out # vanilla lora
                        print(f"[{self._layer_name}] lora: (input @ down @ up)")
                        # print(f"  lora_down: {lora_down.weight.shape}, lora_up: {lora_up.weight.shape}")
                        # print(f"  z (after down) shape: {z.shape}")
                        print(f"  lora_out norm: {lora_out.norm().item():.6f} | mean: {lora_out.mean().item():.6f}")
                        print(f"  output (final) norm: {output.norm().item():.6f} | mean: {output.mean().item():.6f}")

                        # print(
                        #     f"[{self._layer_name}] bit={bit_key} | base: {output.norm():.4f} | lora: {lora_out.norm():.4f}")
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
        elif self.bitwidth_schedule == "random":
            # Use random bit-width assignment (original behavior)
            bit_config = {}
            for name, module in self.model.named_modules():
                if hasattr(module, "_quantized_weights") and "lm_head" not in name:
                    chosen_bit = random.choice(self.bit_choices)
                    bit_config[name] = chosen_bit
            set_active_bitwidths(self.model, bit_config)
        else:
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

# This class serves 2 purposes:
# 1. Workaround to disable gradient updates to the WTE layer while keeping gradient_required = True
# 2. Log grad norms to compare update strengths of WTE vs Lora
class SFTTrainerWithGradLoggingNoWTE(SFTTrainer):
    def training_step(self, model, inputs, num_steps_in_batch):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        print(f"🔥 compute_loss: loss.requires_grad = {loss.requires_grad}")
        print(f"🔥 compute_loss: loss.grad_fn = {loss.grad_fn}")
        loss.backward()

        # 🔍 Grad norm logging
        try:
            wte_norm = model.transformer.wte.weight.grad.norm().item()
            lora_norms = []
            for name, param in model.named_parameters():
                if "lora" in name and param.grad is not None:
                    lora_norms.append(param.grad.norm().item())
            avg_lora_norm = sum(lora_norms) / len(lora_norms) if lora_norms else 0.0
            print(f"🧠 Grad Norms | wte: {wte_norm:.4f} | avg lora: {avg_lora_norm:.4f}")
        except Exception as e:
            print(f"[Grad Logging] Error: {e}")

        return loss.detach()

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Filter out 'wte' from optimizer param groups
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or "wte" in name:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        grouped_params = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(grouped_params, lr=self.args.learning_rate)
        return self.optimizer

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
            bitwidth_schedule=qat_args.bitwidth_schedule,
            cyclic_repeat_per_bit=qat_args.cyclic_repeat_per_bit
        )]
        
        if qat_args.bitwidth_schedule in ["cyclic", "static", "random"]:
            print(f"[Config] Using {qat_args.bitwidth_schedule} bit-width assignment from: {qat_args.bit_choices}")
        else:
            print(f"[Warning] Unknown bitwidth_schedule: {qat_args.bitwidth_schedule}. Defaulting to 'static'")
            qat_args.bitwidth_schedule = "static"
            print(f"[Config] Using {qat_args.bitwidth_schedule} bit-width assignment from: {qat_args.bit_choices}")
        
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
    # trainer = SFTTrainer(
    trainer=SFTTrainerWithGradLoggingNoWTE( # 7/6: allow print grad norm AND prevent wte grad updates
        model=model,
        args=training_args,
        train_dataset=train_dataset, # dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset, # dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks = callbacks
    )

    # Print trainable parameters before training
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

    # Train
    trainer.train()

    # 🔍 Confirm wte is excluded from optimizer
    wte_ref = model.transformer.wte.weight
    wte_found = any(p is wte_ref for g in trainer.optimizer.param_groups for p in g["params"])
    if wte_found:
        print("🚨 wte IS in optimizer! (unexpected)")
    else:
        print("✅ wte is NOT in optimizer — LoRA-only training confirmed.")

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
