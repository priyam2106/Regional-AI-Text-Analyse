"""Train a fast, local Hindi AI-text baseline using character and word TF-IDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline

from train import DATASET_ID, build_examples, group_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="model/detector.joblib")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_examples(args.max_examples)
    train_rows, test_rows = group_split(examples, test_size=0.15, seed=args.seed)
    features = FeatureUnion([
        ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=120000, sublinear_tf=True)),
        ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
    ])
    pipeline = Pipeline([( "features", features), ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0))])
    pipeline.fit([x["text"] for x in train_rows], [x["label"] for x in train_rows])
    predicted = pipeline.predict([x["text"] for x in test_rows])
    labels = [x["label"] for x in test_rows]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    metadata = {"model_type": "character_and_word_tfidf_logistic_regression", "dataset": DATASET_ID, "train_examples": len(train_rows), "test_examples": len(test_rows), "metrics": {"accuracy": accuracy_score(labels, predicted), "precision": precision, "recall": recall, "f1": f1}, "note": "Hindi news-domain baseline; never use as proof of authorship."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, output)
    (output.parent / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
