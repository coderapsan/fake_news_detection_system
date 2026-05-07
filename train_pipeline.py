#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from text_utils import clean_text, normalize_text


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the fake news pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip-path", type=str, help="Path to archive.zip containing Fake.csv and True.csv")
    group.add_argument("--data-dir", type=str, help="Folder containing Fake.csv and True.csv")
    parser.add_argument("--output-dir", type=str, default="masters_project_output", help="Output folder")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional stratified sample size per class for quick experiments. Leave empty for full dataset.",
    )
    parser.add_argument(
        "--best-model-metric",
        type=str,
        default="f1",
        choices=["accuracy", "precision", "recall", "f1"],
        help="Metric used to select the best model.",
    )
    return parser.parse_args()


def load_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        fake = pd.read_csv(zf.open("Fake.csv"))
        true = pd.read_csv(zf.open("True.csv"))
    fake["label"] = 1
    true["label"] = 0
    return pd.concat([fake, true], ignore_index=True)


def load_from_dir(data_dir: Path) -> pd.DataFrame:
    fake = pd.read_csv(data_dir / "Fake.csv")
    true = pd.read_csv(data_dir / "True.csv")
    fake["label"] = 1
    true["label"] = 0
    return pd.concat([fake, true], ignore_index=True)


def load_and_prepare_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if args.zip_path:
        df = load_from_zip(Path(args.zip_path))
    else:
        df = load_from_dir(Path(args.data_dir))

    rows_before = len(df)
    missing_title = int(df["title"].isna().sum())
    missing_text = int(df["text"].isna().sum())

    for col in ["title", "text", "subject", "date"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(normalize_text)

    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    duplicates_removed = int(df.duplicated(subset=["content"]).sum())
    df = df.drop_duplicates(subset=["content"]).copy()

    df["char_len"] = df["content"].str.len()
    short_rows_removed = int((df["char_len"] < 50).sum())
    df = df[df["char_len"] >= 50].copy()

    if args.sample_per_class:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), args.sample_per_class), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )

    summary = {
        "rows_before_cleaning": rows_before,
        "missing_title": missing_title,
        "missing_text": missing_text,
        "duplicates_removed": duplicates_removed,
        "short_rows_removed": short_rows_removed,
        "rows_after_cleaning": int(len(df)),
        "fake_after_cleaning": int((df["label"] == 1).sum()),
        "real_after_cleaning": int((df["label"] == 0).sum()),
        "mean_char_len": float(df["char_len"].mean()),
        "median_char_len": float(df["char_len"].median()),
    }
    return df.reset_index(drop=True), summary


def save_dataset_summary(summary: Dict[str, float], output_dir: Path) -> None:
    out = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    out.to_csv(output_dir / "dataset_summary.csv", index=False)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    df["label"].map({1: "Fake", 0: "Real"}).value_counts().plot(kind="bar")
    plt.title("Class Distribution After Cleaning")
    plt.xlabel("Class")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.hist(df["char_len"], bins=50)
    plt.title("Document Length Distribution")
    plt.xlabel("Character Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "document_length_distribution.png", dpi=220)
    plt.close()

    if "subject" in df.columns:
        plt.figure(figsize=(9, 4))
        df["subject"].value_counts().head(10).plot(kind="bar")
        plt.title("Top Subjects in the Dataset")
        plt.xlabel("Subject")
        plt.ylabel("Count")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "top_subjects.png", dpi=220)
        plt.close()


def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def feature_engineering_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    configs = [
        (
            "text_unigram",
            train_df["text"],
            test_df["text"],
            dict(ngram_range=(1, 1), max_features=30000, min_df=3, max_df=0.95),
        ),
        (
            "text_uni_bigram",
            train_df["text"],
            test_df["text"],
            dict(ngram_range=(1, 2), max_features=50000, min_df=3, max_df=0.95),
        ),
        (
            "title_only_uni_bigram",
            train_df["title"],
            test_df["title"],
            dict(ngram_range=(1, 2), max_features=20000, min_df=2, max_df=0.95),
        ),
        (
            "title_plus_text_uni_bigram",
            train_df["content"],
            test_df["content"],
            dict(ngram_range=(1, 2), max_features=60000, min_df=3, max_df=0.95),
        ),
    ]

    rows = []
    for name, X_train, X_test, vec_params in configs:
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=clean_text,
                        stop_words="english",
                        sublinear_tf=True,
                        **vec_params,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
            ]
        )
        pipe.fit(X_train, train_df["label"])
        pred = pipe.predict(X_test)
        rows.append(
            {
                "config": name,
                "accuracy": accuracy_score(test_df["label"], pred),
                "precision": precision_score(test_df["label"], pred),
                "recall": recall_score(test_df["label"], pred),
                "f1": f1_score(test_df["label"], pred),
            }
        )

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(output_dir / "feature_engineering_results.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.bar(results["config"], results["f1"])
    plt.title("Feature Engineering Comparison")
    plt.xlabel("Feature Configuration")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_engineering_f1.png", dpi=220)
    plt.close()

    return results


def build_model_pipelines(best_config: str) -> Dict[str, dict]:
    config_map = {
        "text_unigram": ("text", dict(ngram_range=(1, 1), max_features=30000, min_df=3, max_df=0.95)),
        "text_uni_bigram": ("text", dict(ngram_range=(1, 2), max_features=50000, min_df=3, max_df=0.95)),
        "title_only_uni_bigram": ("title", dict(ngram_range=(1, 2), max_features=20000, min_df=2, max_df=0.95)),
        "title_plus_text_uni_bigram": ("content", dict(ngram_range=(1, 2), max_features=60000, min_df=3, max_df=0.95)),
    }
    input_column, vec_params = config_map[best_config]
    base_vectorizer = dict(
        preprocessor=clean_text,
        stop_words="english",
        sublinear_tf=True,
        **vec_params,
    )
    models = {
        "Logistic Regression": {
            "input_column": input_column,
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**base_vectorizer)),
                    ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
                ]
            ),
        },
        "Multinomial NB": {
            "input_column": input_column,
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**base_vectorizer)),
                    ("clf", MultinomialNB(alpha=0.1)),
                ]
            ),
        },
        "Linear SVM": {
            "input_column": input_column,
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**base_vectorizer)),
                    ("clf", LinearSVC(C=1.0)),
                ]
            ),
        },
        "Random Forest (SVD)": {
            "input_column": input_column,
            "pipeline": Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**base_vectorizer)),
                    ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),
                    ("norm", Normalizer(copy=False)),
                    ("clf", RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=RANDOM_STATE)),
                ]
            ),
        },
    }
    return models


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, best_config: str, output_dir: Path):
    models = build_model_pipelines(best_config)
    rows = []
    trained = {}

    for model_name, payload in models.items():
        input_col = payload["input_column"]
        pipe = payload["pipeline"]
        pipe.fit(train_df[input_col], train_df["label"])
        pred = pipe.predict(test_df[input_col])

        row = {
            "model": model_name,
            "input_column": input_col,
            "accuracy": accuracy_score(test_df["label"], pred),
            "precision": precision_score(test_df["label"], pred),
            "recall": recall_score(test_df["label"], pred),
            "f1": f1_score(test_df["label"], pred),
            "confusion_matrix": confusion_matrix(test_df["label"], pred).tolist(),
        }
        rows.append(row)
        trained[model_name] = {
            "input_column": input_col,
            "pipeline": pipe,
            "pred": pred,
        }

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(output_dir / "model_comparison_results.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(results["model"], results["f1"])
    plt.title("Model Comparison by F1 Score")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_f1.png", dpi=220)
    plt.close()

    return results, trained


def save_best_model_artifacts(
    model_results: pd.DataFrame,
    trained: Dict[str, Dict[str, object]],
    test_df: pd.DataFrame,
    output_dir: Path,
    artifacts_dir: Path,
    metric: str = "f1",
) -> None:
    best_row = model_results.sort_values(metric, ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = trained[best_name]["pipeline"]
    best_input = trained[best_name]["input_column"]
    best_pred = trained[best_name]["pred"]

    joblib.dump(best_model, artifacts_dir / "best_model.joblib", protocol=4)

    report = classification_report(test_df["label"], best_pred, output_dict=True)
    with open(output_dir / "best_model_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metadata = {
        "best_model_name": best_name,
        "best_metric": metric,
        "input_column": best_input,
        "selection_scores": best_row.to_dict(),
        "note": "Linear SVM frequently performs strongly on sparse TF-IDF text data. Random Forest is evaluated after SVD because raw sparse TF-IDF is inefficient for tree ensembles.",
    }
    with open(artifacts_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    cm = np.array(confusion_matrix(test_df["label"], best_pred))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real", "Fake"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / "best_model_confusion_matrix.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    artifacts_dir = output_dir / "artifacts"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df, summary = load_and_prepare_dataframe(args)
    save_dataset_summary(summary, results_dir)
    save_eda_plots(df, results_dir)

    train_df, test_df = split_data(df)

    feature_results = feature_engineering_experiment(train_df, test_df, results_dir)
    best_feature_config = feature_results.iloc[0]["config"]

    model_results, trained = evaluate_models(train_df, test_df, best_feature_config, results_dir)
    save_best_model_artifacts(
        model_results=model_results,
        trained=trained,
        test_df=test_df,
        output_dir=results_dir,
        artifacts_dir=artifacts_dir,
        metric=args.best_model_metric,
    )

    run_summary = {
        "feature_engineering_best_config": best_feature_config,
        "best_model_by_metric": args.best_model_metric,
        "top_model": model_results.sort_values(args.best_model_metric, ascending=False).iloc[0]["model"],
        "sample_per_class": args.sample_per_class,
        "rows_used": int(len(df)),
    }

    with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("Pipeline complete.")
    print(f"Outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()