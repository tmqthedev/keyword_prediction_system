import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "final_model"


def load_and_prepare_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_FILE}")

    loaded_dataset = cast(DatasetDict, load_dataset("csv", data_files={"train": str(DATA_FILE)}))
    train_split = cast(Dataset, loaded_dataset["train"])

    required_columns = {"query", "keyword"}
    missing = required_columns.difference(set(train_split.column_names))
    if missing:
        raise ValueError(f"Missing required columns in data.csv: {sorted(missing)}")

    # Create train/test split from a single source file.
    dataset = cast(DatasetDict, train_split.train_test_split(test_size=0.2, seed=42))
    print("Loaded dataset successfully")
    return dataset, tokenizer


def preprocess_data(dataset, tokenizer):
    all_keywords = list(dataset["train"]["keyword"]) + list(dataset["test"]["keyword"])
    keywords = sorted(set(all_keywords))
    label_to_id = {label: idx for idx, label in enumerate(keywords)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    def tokenize_and_encode(examples):
        tokenized = tokenizer(
            examples["query"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tokenized["labels"] = [label_to_id[k] for k in examples["keyword"]]
        return tokenized

    tokenized_datasets = cast(DatasetDict, dataset.map(
        tokenize_and_encode,
        batched=True,
        remove_columns=dataset["train"].column_names,
    ))
    tokenized_datasets.set_format(type="torch")
    print("Tokenization and label encoding completed")
    return tokenized_datasets, label_to_id, id_to_label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def setup_training(tokenized_datasets, label_to_id, id_to_label, tokenizer):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label=id_to_label,
    )
    if isinstance(model, tuple):
        model = model[0]

    tokenized = cast(Any, tokenized_datasets)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=str(BASE_DIR / "logs"),
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


def save_label_mapping(label_to_id, id_to_label):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = MODEL_DIR / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as mapping_file:
        id_to_label_for_json: Dict[str, str] = {str(k): v for k, v in id_to_label.items()}
        json.dump(
            {"label_to_id": label_to_id, "id_to_label": id_to_label_for_json},
            mapping_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved label mapping to {mapping_path}")


def main():
    dataset, tokenizer = load_and_prepare_data()
    tokenized_datasets, label_to_id, id_to_label = preprocess_data(dataset, tokenizer)

    print("\nDataset Info:")
    print(f"Train set size: {len(tokenized_datasets['train'])}")
    print(f"Test set size: {len(tokenized_datasets['test'])}")
    print(f"Number of labels: {len(label_to_id)}")

    trainer = setup_training(tokenized_datasets, label_to_id, id_to_label, tokenizer)

    print("\nStarting training...")
    trainer.train()
    print("Training completed successfully")

    trainer.save_model(str(MODEL_DIR))
    save_label_mapping(label_to_id, id_to_label)
    print(f"Model saved successfully to {MODEL_DIR}")


if __name__ == "__main__":
    main()