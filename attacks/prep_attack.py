import json

def simple_perturbation(prompt):
    swaps = {
        "who": "whom",
        "what": "wut",
        "where": "wher",
        "when": "whn",
        "how": "hw",
        "why": "whi",
        "is": "iz",
        "the": "teh",
        "a": "uh",
        "and": "&"
    }
    words = prompt.split()
    perturbed_words = []
    changes = []
    for w in words:
        w_pert = swaps.get(w.lower(), w)
        if w.lower() != w_pert:
            changes.append((w, w_pert))
        perturbed_words.append(w_pert)
    return " ".join(perturbed_words), changes

# Paths
input_json_path = "/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set.json"
output_json_path = "/content/drive/MyDrive/Colab_Notebooks/eic_llm/eval_set_perturbed.json"

# Load original
with open(input_json_path, "r") as f:
    lines = [json.loads(l) for l in f]

# Perturb and print first example
first = lines[0]
context_orig = first["context"]
question_orig = first["question"]

context_pert, context_changes = simple_perturbation(context_orig)
question_pert, question_changes = simple_perturbation(question_orig)

print("📘 Original Context:\n", context_orig)
print("🛠️ Perturbed Context:\n", context_pert)
print("🔄 Changed in Context:", context_changes)

print("\n❓ Original Question:\n", question_orig)
print("🛠️ Perturbed Question:\n", question_pert)
print("🔄 Changed in Question:", question_changes)

# Save full perturbed dataset
perturbed_lines = []
for ex in lines:
    new_ex = ex.copy()
    new_ex["context"], _ = simple_perturbation(ex["context"])
    new_ex["question"], _ = simple_perturbation(ex["question"])
    perturbed_lines.append(new_ex)

with open(output_json_path, "w") as f:
    for ex in perturbed_lines:
        f.write(json.dumps(ex) + "\n")

print(f"\n✅ Wrote perturbed dataset to {output_json_path}")
