import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm

# Paths
model_path = "/content/drive/MyDrive/Colab_Notebooks/gpt2-sft"  # <-- UPDATE THIS
eval_json_path = "/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json"

# Load tokenizer and trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
model.eval()

# Load validation examples from JSON
with open(eval_json_path, "r") as f:
    dataset = [json.loads(line) for line in f]

# Load SQuAD metric
metric = evaluate.load("squad")

# Inference loop
predictions, references = [], []

for example in tqdm(dataset, desc="Evaluating"):
    context = example["context"].strip()
    question = example["question"].strip()
    qid = example.get("id", f"id_{len(predictions)}")
    prompt = f"{example['context'].strip()}\n{example['question'].strip()}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    predictions.append({
        "id": qid,
        "prediction_text": generated
    })

    references.append({
        "id": qid,
        "answers": example["answers"]
    })

# Compute EM and F1
results = metric.compute(predictions=predictions, references=references)
print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")
