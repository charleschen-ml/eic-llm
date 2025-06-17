# Load dataset from drive, create split, upload to HF

if __name__ == "__main__":
    import pandas as pd

    # Load dataset from drive
    from datasets import Dataset, DatasetDict
    df = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/my_dataset/sft_dataset.csv")
    df["text"] = df["text_instr"] + "\n" + df["text_input"] + "\n" + df["text_label"]
    df["text_prompt"] = df["text_instr"] + "\n" + df["text_input"] + "\n"

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create train/test/val split: 80/10/10
    train_end = int(0.8 * len(df))
    val_end = int(0.9 * len(df))
    ds = DatasetDict({
        "train": Dataset.from_pandas(df[:train_end]),
        "val": Dataset.from_pandas(df[train_end:val_end]),
        "test": Dataset.from_pandas(df[val_end:])
    })

    # Upload to HF
    ds.push_to_hub("shoubing35/ones_digit_sft_dataset")

