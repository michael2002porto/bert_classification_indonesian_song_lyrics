import torch
import numpy as np
from transformers import BertTokenizer
import shap
from lime.lime_text import LimeTextExplainer

from models.multi_class_model import MultiClassModel

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Load final checkpoint
model = MultiClassModel.load_from_checkpoint(
    "final_checkpoints/original_split_synthesized.ckpt",
    n_out=4,
    dropout=0.3,
    lr=1e-5
)

# Ensure model is in evaluation mode
model.eval()

# Define test lyric ["semua usia", "anak", "remaja", "dewasa"]
test_lyric = "hidup ini adalah kesempatan. hidup ini untuk melayani tuhan. jangan sia-siakan waktu yang tuhan bri"

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

# Perform inference
with torch.no_grad():
    test_prediction = model(
        encoding["input_ids"],
        encoding["attention_mask"],
        encoding["token_type_ids"]
    )

probabilities = torch.nn.functional.softmax(test_prediction, dim=1).cpu().numpy().flatten()

age_groups = ["semua usia", "anak", "remaja", "dewasa"]
predicted_class = np.argmax(probabilities)

# Print prediction results
print("==== Age Group Prediction ====")
print(f"Input Lyric: {test_lyric}")
print(f"Predicted Age Group: {age_groups[predicted_class]}")
print("\nClass Probabilities:")
for idx, label in enumerate(age_groups):
    print(f"{label}: {probabilities[idx]:.4f}")

# SHAP Explanation with proper tokenization handling
def custom_tokenizer(text):
    return tokenizer.tokenize(text, add_special_tokens=True)

def custom_detokenizer(tokens):
    return tokenizer.convert_tokens_to_string(tokens)

# Create SHAP masker with BERT-compatible settings
masker = shap.maskers.Text(
    tokenizer=custom_tokenizer,
    mask_token=tokenizer.mask_token,
    collapse_mask_token=False,
    output_type="string",
    decoder=custom_detokenizer
)

# Create explainer with padding handling
def padded_model_wrapper(texts):
    # Tokenize with padding
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        truncation=True
    )
    
    # Move to model device
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    token_type_ids = encodings["token_type_ids"].to(model.device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    return outputs.cpu().numpy()

explainer_shap = shap.Explainer(
    padded_model_wrapper,
    masker,
    output_names=age_groups
)

# Generate explanation
shap_values = explainer_shap([test_lyric])

print("\n==== SHAP Explanation ====")
shap.plots.text(shap_values[:, :, predicted_class], display=False)

# LIME Explanation
def predict_proba(texts):
    encodings = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        encodings.append(encoding)
    
    input_ids = torch.cat([e["input_ids"] for e in encodings]).to(model.device)
    attention_mask = torch.cat([e["attention_mask"] for e in encodings]).to(model.device)
    token_type_ids = torch.cat([e["token_type_ids"] for e in encodings]).to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

explainer_lime = LimeTextExplainer(class_names=age_groups)
exp = explainer_lime.explain_instance(test_lyric, predict_proba, num_features=10, num_samples=500)

print("\n==== LIME Explanation ====")
print("Top features contributing to the prediction:")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")

# CEM Note
print("\n==== CEM Consideration ====")
print("Counterfactual Explanations (CEM) require additional setup such as a trained autoencoder")
print("or generative model to produce meaningful text counterfactuals. Consider using libraries")
print("like ALIBI or TextAttack for advanced implementations.")