import torch
import numpy as np
from transformers import BertTokenizer

from models.multi_class_model import MultiClassModel

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Load final checkpoint
model = MultiClassModel.load_from_checkpoint(
    "final_checkpoints/original_split_synthesized.ckpt",
    n_out = 4,
    dropout = 0.3,  # dropout tentuin sendiri
    lr = 1e-5
)

# Ensure model is in evaluation mode
model.eval()

# Define test lyric ["semua usia", "anak", "remaja", "dewasa"]
test_lyric = "hidup ini adalah kesempatan. hidup ini untuk melayani tuhan. jangan sia-siakan waktu yang tuhan bri"
# test_lyric = "Aku adalah anak gembalaSelalu riang serta gembiraKarena aku senang bekerjaTak pernah malas atau pun lengah"
# test_lyric = "Kala kupandang kerlip bintang nun jauh disanaSaat kudenger melodi cinta yang menggemaTerasa kembali gelora jiwa mudakuKarna tersentuh alunan lagu semerdu kopi dangdutApi asmara yang dahulu pernah membaraSemakin hangat bagai ciuman yang pertamaDetak jantungku seakan ikut iramaKarna terlena oleh pesona alunan kopi dangdut"
# test_lyric = "Oh hip hip hura hura (hura hura)Aku suka dia (suka dia)Aku jatuh cinta (jatuh cinta)Dia menanti cinta bersemi di hati"

# Encode text for BERT
encoding = tokenizer.encode_plus(
    test_lyric,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=True,
    padding="max_length",
    return_attention_mask=True,
    return_tensors='pt',
)

# Perform inference without gradient calculation
with torch.no_grad():
    test_prediction = model(
        encoding["input_ids"],
        encoding["attention_mask"],
        encoding["token_type_ids"]
    )

# ✅ Fix: Use `torch.nn.functional.softmax()` instead of `torch.nn.Softmax()`
probabilities = torch.nn.functional.softmax(test_prediction, dim=1).cpu().numpy().flatten()

# Define age group labels
age_groups = ["semua usia", "anak", "remaja", "dewasa"]

# Get predicted class index
predicted_class = np.argmax(probabilities)

# Print output
print("==== Age Group Prediction ====")
print(f"Input Lyric: {test_lyric}")
print(f"Predicted Age Group: {age_groups[predicted_class]}")
print("\nClass Probabilities:")
for idx, label in enumerate(age_groups):
    print(f"{label}: {probabilities[idx]:.4f}")
