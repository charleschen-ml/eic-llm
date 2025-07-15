# Inference using 1.) base model, 2.) sft-trained model

import shutil
import evaluate
import json
import csv
import argparse
from math import ceil
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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # To fix torch deterministic error
torch.use_deterministic_algorithms(True)
from textattack.models.wrappers import ModelWrapper
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

# Custom arguments for inference-specific parameters
class InferenceArguments:
    def __init__(self, 
                 eval_json_path="/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json",
                 adapter_path="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat",
                 output_csv_path="/content/drive/MyDrive/Colab_Notebooks/eic_llm/inference_output.csv",
                 bitwise_lora_adapter_path="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat/full_qat_model.pt",
                 use_quantization=True,
                 use_bitwise_lora=True,
                 bit_choices="32",
                 max_inf_size=100,
                 quant_layers="6,11",
                 inf_bit_config={},
                 default_bit=32):
        self.eval_json_path = eval_json_path
        self.adapter_path = adapter_path
        self.output_csv_path = output_csv_path
        self.bitwise_lora_adapter_path = bitwise_lora_adapter_path
        self.use_quantization = use_quantization
        self.use_bitwise_lora = use_bitwise_lora
        self.bit_choices = bit_choices
        self.max_inf_size = max_inf_size
        self.quant_layers = quant_layers
        self.inf_bit_config = inf_bit_config
        self.default_bit = default_bit

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
def save_predictions_to_csv(predictions, references, output_csv_path):
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

    with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prediction", "reference", "exact_match", "f1_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} results to {output_csv_path}")

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, PPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("inference", help="Run the inference script", dataclass_types=dataclass_types)
    else:
        parser = HfArgumentParser(dataclass_types)
    
    # Add inference-specific arguments
    parser.add_argument("--eval_json_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json",
                       help="Path to evaluation JSON file")
    parser.add_argument("--adapter_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat",
                       help="Path to adapter directory")
    parser.add_argument("--output_csv_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/eic_llm/inference_output.csv",
                       help="Path to save inference output CSV")
    parser.add_argument("--bitwise_lora_adapter_path", type=str, 
                       default="/content/drive/MyDrive/Colab_Notebooks/gpt2-qat/full_qat_model.pt",
                       help="Path to bitwise LoRA adapter file")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                       help="Whether to apply quantization")
    parser.add_argument("--no_quantization", dest="use_quantization", action="store_false",
                       help="Disable quantization")
    parser.add_argument("--use_bitwise_lora", action="store_true", default=True,
                       help="Whether to use bitwise LoRA adapters")
    parser.add_argument("--no_bitwise_lora", dest="use_bitwise_lora", action="store_false",
                       help="Disable bitwise LoRA adapters")
    parser.add_argument("--bit_choices", type=str, default="32",
                       help="Comma-separated list of bit choices for LoRA")
    parser.add_argument("--max_inf_size", type=int, default=100,
                       help="Maximum number of examples to infer")
    parser.add_argument("--quant_layers", type=str, default="6,11",
                       help="Comma-separated list of h.* layers to quantize")
    parser.add_argument("--inf_bit_config", type=str, default=None,
                       help="JSON string for inference bit configuration (e.g., '{\"transformer.h.0\": 8, \"transformer.h.1\": 4}'). Default: 32 bits for all layers")
    parser.add_argument("--default_bit", type=int, default=32,
                       help="Default bit for all layers")
    return parser

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip().split("\n")[0].strip()

def run_adverse(model, tokenizer, dataset):
    # Create inputs and references
    inputs = []
    references = []
    for example in tqdm(dataset, desc="Evaluating", disable=True):
        context = example["context"].strip()
        question = example["question"].strip()
        qid = example.get("id", f"id_{len(inputs)}")
        prompt = f"{context}\n{question}"
        answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
        inputs.append((prompt, answer))

        references.append({
            "id": qid,
            "answers": example["answers"]
        })
    print("\n[run_adverse] inputs and refs created")

    # Create attacker
    attack_dataset = Dataset([(prompt, 1) for prompt, _ in inputs])
    NUM_EXAMPLES = len(inputs)
    model_wrapper = DummyClassificationWrapper(tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)
    attack_args = AttackArgs(num_examples=NUM_EXAMPLES, disable_stdout=True)
    attacker = Attacker(attack, attack_dataset, attack_args)
    print("\n[run_adverse] attacker created")

    # Inference loop
    predictions_orig = []
    predictions_pert = []
    for i, (attack_result, (prompt, ground_truth)) in enumerate(zip(attacker.attack_dataset(), inputs)):
        orig_prompt = attack_result.original_text()
        print(f"orig_prompt = {orig_prompt}")
        pert_prompt = attack_result.perturbed_text()
        print(f"pert_prompt = {pert_prompt}")
        qid = f"id_{i}"

        pred_orig = generate_answer(model, tokenizer, orig_prompt)
        pred_pert = generate_answer(model, tokenizer, pert_prompt)

        predictions_orig.append({
            "id": qid,
            "prediction_text": pred_orig
        })

        predictions_pert.append({
            "id": qid,
            "prediction_text": pred_pert
        })

    print("\nScoring original predictions:")
    results_orig = score_squad(predictions_orig, references)
    save_predictions_to_csv(predictions_orig, references, "/content/drive/MyDrive/Colab_Notebooks/eic_llm/predictions_orig.csv")

    print("\nScoring perturbed predictions:")
    results_pert = score_squad(predictions_pert, references)
    save_predictions_to_csv(predictions_pert, references, "/content/drive/MyDrive/Colab_Notebooks/eic_llm/predictions_pert.csv")

class DummyClassificationWrapper(ModelWrapper):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = self

    def __call__(self, text_input_list):
        # Just tokenize to appease TextAttack internals
        _ = self.tokenizer(text_input_list, return_tensors="pt", padding=True, truncation=True)
        return [[0.1, 0.9] for _ in text_input_list]

def main(script_args, training_args, model_args, inference_args):
    """
    Main inference function with configurable parameters.
    
    Args:
        script_args, training_args, model_args: Standard TRL arguments
        inference_args: InferenceArguments object containing inference-specific parameters
    """
    # Convert string arguments to lists if they're strings
    if isinstance(inference_args.bit_choices, str):
        bit_choices = [int(x.strip()) for x in inference_args.bit_choices.split(",")]
    else:
        bit_choices = inference_args.bit_choices
        
    if isinstance(inference_args.quant_layers, str):
        quant_layers = [int(x.strip()) for x in inference_args.quant_layers.split(",")]
    else:
        quant_layers = inference_args.quant_layers
    
    
    # Load validation examples from JSON
    with open(inference_args.eval_json_path, "r") as f:
        dataset = [json.loads(line) for line in f][:inference_args.max_inf_size]
    print(f"Examples used for inference: {len(dataset)}")

    # remove output_dir if exists
    if training_args is not None:
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
    if inference_args.use_quantization:
        patch_linear_forward_with_switchable_quantization(base_model, bit_widths=bit_choices, quant_layers=quant_layers)
        add_bitwise_lora_adapters(base_model, bit_widths=bit_choices, quant_layers=quant_layers)

        # Dummy forward to create LoRA modules, which are created at run time to fix matrix dim mismatch error
        dummy_input = tokenizer("hello", return_tensors="pt").to(base_model.device)
        _ = base_model(**dummy_input)

        state_dict = torch.load(inference_args.bitwise_lora_adapter_path, map_location="cpu")
        base_model.load_state_dict(state_dict)
        base_model.to("cuda")
        set_active_bitwidths(base_model, inference_args.inf_bit_config, inference_args.default_bit)
        base_model.eval()

    # load squad dataset from hf
    # df = load_dataset("rajpurkar/squad", split="train") # split="train" or "validation"
    # USE_ADVERSE = True
    # if USE_ADVERSE:
    #     df_adverse = df.map(lambda x: { # list of tuples (prompt, answer)
    #         "prompt": x["context"].strip() + "\n" + x["question"].strip(),
    #         "answer": x["answers"]["text"][0] if x["answers"]["text"] else ""
    #     })
    #     inputs = list(zip(df_adverse["prompt"], df_adverse["answer"]))
    #     run_adverse(base_model, tokenizer, inputs) # run adversarial EM eval
    #
    # df_context = df["context"]
    # df_question = df["question"]
    # df = df.map(lambda x: {"prompt": x["context"].strip() + "\n" + x["question"].strip() + "\n"}) # match sft style

    ################
    # Generate completions after sft training
    ################

    # Load sft-trained peft model
    if inference_args.use_bitwise_lora:
        peft_sft = base_model # use base model for bitwise lora
    else:
        peft_sft = PeftModel.from_pretrained(base_model, inference_args.adapter_path)  # Load peft model
        peft_sft.eval()

    # Print created lora
    print("\n🔍 Created LoRA Adapters:")
    for name, module in peft_sft.named_modules():
        if hasattr(module, "_lora_adapters"):
            for bw, lora in module._lora_adapters.items():
                weights = list(lora.parameters())
                norm = sum(p.norm().item() for p in weights)
                print(f"{name} | {bw}-bit LoRA norm: {norm:.4f}")

    # Print active lora
    print("\n🔍 Active LoRA Adapters:")
    for name, module in peft_sft.named_modules():
        if hasattr(module, "_lora_adapters") and hasattr(module, "_active_bit"):
            print(f"{name} | Active bitwidth: {module._active_bit} | Available: {list(module._lora_adapters.keys())}")

    print("\nAdversarial attack:\n")

    # Run adversarial attack
    run_adverse(base_model, tokenizer, dataset)

    return 0

if __name__ == "__main__":
    # parse script arguments
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Parse inference-specific arguments
    args = parser.parse_args()
    
    # Convert string arguments to lists
    bit_choices = [int(x.strip()) for x in args.bit_choices.split(",")]
    quant_layers = [int(x.strip()) for x in args.quant_layers.split(",")]
    
    # Create inference arguments object
    inference_args = InferenceArguments(
        eval_json_path=args.eval_json_path,
        adapter_path=args.adapter_path,
        output_csv_path=args.output_csv_path,
        bitwise_lora_adapter_path=args.bitwise_lora_adapter_path,
        use_quantization=args.use_quantization,
        use_bitwise_lora=args.use_bitwise_lora,
        bit_choices=args.bit_choices,
        max_inf_size=args.max_inf_size,
        quant_layers=args.quant_layers,
        inf_bit_config=args.inf_bit_config,
        default_bit=args.default_bit
    )
    
    # Run inference
    main(script_args, training_args, model_args, inference_args)
