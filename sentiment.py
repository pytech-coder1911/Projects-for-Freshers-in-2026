# ============================================================
# PROJECT 3: BERT Sentiment Analyzer
# Stack: HuggingFace Transformers, PyTorch, Gradio
# Install: pip install transformers datasets torch gradio
# Deploy free: https://huggingface.co/spaces
# ============================================================

# ---------- Part A: Fine-tune DistilBERT ----------
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
import numpy as np
import evaluate
import pandas as pd
import torch

MODEL_NAME  = "distilbert-base-uncased"
NUM_LABELS  = 3   # negative=0, neutral=1, positive=2
OUTPUT_DIR  = "./sentiment_model"

tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
accuracy    = evaluate.load("accuracy")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

def train_from_csv(csv_path: str):
    """
    CSV must have columns: text, label (0=negative, 1=neutral, 2=positive)
    Example: Amazon product reviews, Twitter sentiment datasets
    """
    df      = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df[["text", "label"]])
    split   = dataset.train_test_split(test_size=0.2, seed=42)

    tokenized = split.map(tokenize_fn, batched=True)
    collator  = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label={0:"negative", 1:"neutral", 2:"positive"},
        label2id={"negative":0, "neutral":1, "positive":2}
    )

    args = TrainingArguments(
        output_dir            = OUTPUT_DIR,
        num_train_epochs      = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        evaluation_strategy   = "epoch",
        save_strategy         = "epoch",
        load_best_model_at_end= True,
        metric_for_best_model = "accuracy",
        logging_steps         = 50,
        warmup_steps          = 100,
        weight_decay          = 0.01,
        report_to             = "none"
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = tokenized["train"],
        eval_dataset    = tokenized["test"],
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = compute_metrics
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


# ---------- Part B: Inference ----------
from transformers import pipeline

def load_predictor(model_path: str = OUTPUT_DIR):
    return pipeline("text-classification", model=model_path,
                    tokenizer=model_path, return_all_scores=True)

def predict(text: str, predictor=None):
    if predictor is None:
        predictor = load_predictor()
    scores = predictor(text)[0]
    top    = max(scores, key=lambda x: x["score"])
    return {
        "label"     : top["label"],
        "confidence": round(top["score"] * 100, 2),
        "all_scores": {s["label"]: round(s["score"], 4) for s in scores}
    }


# ---------- Part C: Gradio Demo ----------
import gradio as gr

def gradio_app():
    predictor = load_predictor()
    LABELS    = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    COLORS    = {"positive": "green",    "neutral": "orange",  "negative": "red"}

    def analyze(text):
        if not text.strip():
            return "Please enter some text.", None
        result = predict(text, predictor)
        label  = result["label"]
        conf   = result["confidence"]
        scores = result["all_scores"]
        output = f"**Sentiment:** {LABELS[label]}\n**Confidence:** {conf}%"
        chart  = {
            "label": list(scores.keys()),
            "score": list(scores.values())
        }
        return output, pd.DataFrame(chart)

    demo = gr.Interface(
        fn      = analyze,
        inputs  = gr.Textbox(lines=5, placeholder="Paste a product review or comment here..."),
        outputs = [gr.Markdown(), gr.DataFrame(label="Score Breakdown")],
        title   = "Sentiment Analyzer — BERT",
        description = "Fine-tuned DistilBERT for 3-class sentiment (Positive / Neutral / Negative)",
        examples = [
            ["The product quality is absolutely amazing, totally worth the price!"],
            ["Delivery was okay but packaging was a bit damaged."],
            ["Terrible experience. Product stopped working after 2 days. Avoid!"]
        ]
    )
    demo.launch(share=True)  # share=True gives a public URL


if __name__ == "__main__":
    # To train:   comment out gradio_app() and call train_from_csv("your_data.csv")
    # To demo:    call gradio_app() after model is trained
    gradio_app()
