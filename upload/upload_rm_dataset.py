##########
# WIP
##########

# Process preference CSV, Extract Chosen & Rejected

from datasets import Dataset
import pandas as pd

def process_annotations_and_push_to_hub(csv_path, dataset_name):
    print(f"csv_path = {csv_path}")  # charles
    df = pd.read_csv(csv_path)
    df.head() # charles
    rows = []

    for _, row in df.iterrows():
        if row["preference"] not in ("A", "B"):
            continue  # Skip unannotated rows

        system_entry = {"content": row["system"], "role": "system"}
        prompt_entry = {"content": row["prompt"], "role": "user"}
        response_a = {"content": row["A_response"], "role": "assistant"}
        response_b = {"content": row["B_response"], "role": "assistant"}

        if row["preference"] == "A":
            chosen = [response_a]
            rejected = [response_b]
        else:
            chosen = [response_b]
            rejected = [response_a]

        rows.append({"prompt": [system_entry, prompt_entry], "chosen": chosen, "rejected": rejected})

    # Let Hugging Face infer the schema
    dataset = Dataset.from_list(rows)
    print("\n📊 Dataset preview (first few rows):") # debug
    print(dataset[:3])  # debug

    # Push to Hugging Face Hub
    dataset.push_to_hub(dataset_name)
    print(f"✅ Dataset pushed to: https://huggingface.co/datasets/{dataset_name}")

# call process-and-push function defined above
process_annotations_and_push_to_hub(
    csv_path="/content/drive/MyDrive/Colab_Notebooks/my_dataset/pairwise_comparisons_labeled.csv",
    dataset_name="shoubing35/ones_digit_dataset"
)
