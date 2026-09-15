"""Fine-tune a Hindi AI-text detector on CT² AG_hi.

Labels: 0 = human-written, 1 = AI-generated. The split is grouped by original
article so an article and its generated rewrites never leak across train/test.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

DATASET_ID = "ishank31/CT2_AG_hi"
MODEL_ID = "FacebookAI/xlm-roberta-base"
AI_COLUMNS = ["BARD responses", "GPT3.5_responses", "GPT4_responses", "Gemma_2B_Response", "Gemma_7B_Response"]


def clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value if len(value) >= 80 else None


def build_examples(limit: int | None) -> list[dict]:
    """Convert one human article + several model outputs into binary examples."""
    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
    parquet_files = [name for name in files if name.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError("The dataset repository did not expose Parquet files.")
    examples: list[dict] = []
    for filename in parquet_files:
        local_path = hf_hub_download(DATASET_ID, filename, repo_type="dataset")
        split_name = Path(filename).stem
        for row_id, row in pd.read_parquet(local_path).iterrows():
            group = f"{split_name}-{row_id}"
            # Dataset releases have used both `Article` and `article`, and one
            # Gemma field has a leading space. Normalise headers defensively.
            columns = {str(column).strip().lower(): column for column in row.index}
            headline = clean(row.get(columns.get("headline"), "")) or ""
            human = clean(row.get(columns.get("article")))
            if human:
                examples.append({"text": f"{headline}\n\n{human}", "label": 0, "group": group, "source": split_name})
            for column in AI_COLUMNS:
                generated = clean(row.get(columns.get(column.strip().lower())))
                if generated:
                    examples.append({"text": f"{headline}\n\n{generated}", "label": 1, "group": group, "source": split_name})
    random.shuffle(examples)
    if not limit:
        return examples
    # A small smoke-test sample must still contain both labels. The source data
    # has several generated variants for every human article, so naïve slicing
    # can otherwise produce an all-AI subset.
    human = [example for example in examples if example["label"] == 0]
    generated = [example for example in examples if example["label"] == 1]
    human_count = min(len(human), limit // 2)
    generated_count = min(len(generated), limit - human_count)
    selected = human[:human_count] + generated[:generated_count]
    random.shuffle(selected)
    return selected


def group_split(examples: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    # Split at the original-article level while reserving groups from *each*
    # class. This makes small smoke-test subsets valid as well as the full set.
    label_groups = {0: set(), 1: set()}
    for example in examples:
        label_groups[example["label"]].add(example["group"])
    rng = random.Random(seed)
    test_groups = set()
    for groups in label_groups.values():
        groups = list(groups)
        rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError("Need at least two source articles for each label to make a grouped split.")
        test_groups.update(groups[:max(1, round(len(groups) * test_size))])
    train = [example for example in examples if example["group"] not in test_groups]
    test = [example for example in examples if example["group"] in test_groups]
    if {x["label"] for x in train} != {0, 1}:
        raise ValueError("Grouped split did not retain both labels in training.")
    return train, test


class TextDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        encoded["labels"] = row["label"]
        return encoded


def metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, predictions), "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-examples", type=int, default=None, help="Use a small number for a smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = build_examples(args.max_examples)
    if not examples:
        raise ValueError("No usable examples were found in the dataset.")
    train_rows, test_rows = group_split(examples, test_size=0.15, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_data = TextDataset(train_rows, tokenizer, args.max_length)
    test_data = TextDataset(test_rows, tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2, id2label={0: "HUMAN", 1: "AI"}, label2id={"HUMAN": 0, "AI": 1})
    training_args = TrainingArguments(output_dir=str(output / "checkpoints"), learning_rate=2e-5, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, fp16=torch.cuda.is_available(), report_to="none", logging_steps=25)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=test_data, processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    metadata = {"base_model": MODEL_ID, "dataset": DATASET_ID, "labels": {"0": "human", "1": "ai_generated"}, "train_examples": len(train_rows), "test_examples": len(test_rows), "metrics": evaluation, "note": "Validation is split by source article. Results are not evidence of authorship."}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
