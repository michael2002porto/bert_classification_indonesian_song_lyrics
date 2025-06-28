import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer
from models.multi_class_model import MultiClassModel

label_map = {"anak": 1, "remaja": 2, "dewasa": 3, "semua usia": 0}
label_names = ["semua usia", "anak", "remaja", "dewasa"]

tokenizer = BertTokenizer.from_pretrained("indolem/indobert-base-uncased")


def split_by_capital(text):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def clean_str(string):
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(string)


def get_dataset(path="data/dataset_lyrics.xlsx"):
    df = pd.read_excel(path)
    df = df[["Title", "Lyric", "Age Class tag"]]
    df["Age Class tag"] = df["Age Class tag"].map(label_map)
    df["Lyric"] = df["Lyric"].astype(str).apply(split_by_capital).apply(clean_str)
    return df


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - IndoBERT Multiclass Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_manual_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Expert Judgement")
    plt.xlabel("Predicted Label (Prediksi Sistem)")
    plt.ylabel("True Label (Expert Judgement)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, df):
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        lyric, label = row["Lyric"], row["Age Class tag"]
        if pd.isnull(lyric) or label is None:
            continue

        encoding = tokenizer.encode_plus(
            lyric,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                encoding["input_ids"],
                encoding["attention_mask"],
                encoding["token_type_ids"],
            )
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

    return y_true, y_pred


if __name__ == "__main__":
    # # Load model
    # model = MultiClassModel.load_from_checkpoint(
    #     "final_checkpoints/original_split_synthesized.ckpt",
    #     n_out=4,
    #     dropout=0.3,
    #     lr=1e-5,
    # )
    # model.eval()

    # # Load and preprocess data
    # df = get_dataset("data/dataset_lyrics.xlsx")

    # # Evaluate and plot confusion matrix
    # y_true, y_pred = evaluate_model(model, df)
    # plot_confusion_matrix(y_true, y_pred)

    # Contoh prediksi sistem dan label sebenarnya dari expert judgement
    y_true = [
        0,              # semua usia → anak
        0, 0, 0, 0, 0,  # semua usia → semua usia (5×)
        1, 1, 1,        # anak → anak (3×)
        1,              # anak → dewasa
        2, 2, 2, 2,     # remaja → remaja (4×)
        3,              # dewasa → anak
        3,              # dewasa → remaja
        3, 3, 3, 3,     # dewasa → dewasa (4×)
    ]

    y_pred = [
        1,              # semua usia → anak
        0, 0, 0, 0, 0,  # semua usia → semua usia (5×)
        1, 1, 1,        # anak → anak (3×)
        3,              # anak → dewasa
        2, 2, 2, 2,     # remaja → remaja (4×)
        1,              # dewasa → anak
        2,              # dewasa → remaja
        3, 3, 3, 3,     # dewasa → dewasa (4×)
    ]

    plot_manual_confusion_matrix(y_true, y_pred)
