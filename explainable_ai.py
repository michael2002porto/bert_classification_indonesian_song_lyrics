import torch
import numpy as np
from transformers import BertTokenizer
import shap
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from models.multi_class_model import MultiClassModel


# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Load model on GPU if available
model = MultiClassModel.load_from_checkpoint(
    "final_checkpoint/original_split_synthesized.ckpt",
    n_out=4,
    dropout=0.3,
    lr=1e-5
).to(device).eval()

# Define test lyric and age groups
test_lyric = "Oh hip hip hura hura hura hura Aku suka dia suka dia Aku jatuh cinta jatuh cinta Dia menanti cinta bersemi di hati"
age_groups = ["semua usia", "anak", "remaja", "dewasa"]

# GPU-optimized prediction function
def predict(texts, batch_size=4):  # Increased batch size for GPU
    if isinstance(texts, str):
        texts = [texts]
    
    # Process in batches
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        encodings = tokenizer(
            batch,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(
                encodings["input_ids"],
                encodings["attention_mask"],
                encodings["token_type_ids"]
            )
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    return np.concatenate(all_probs)

# Get prediction
probabilities = predict(test_lyric)[0]
predicted_class = np.argmax(probabilities)

# Print results
print("==== Age Group Prediction ====")
print(f"Predicted Age Group: {age_groups[predicted_class]}")
print("\nClass Probabilities:")
for idx, label in enumerate(age_groups):
    print(f"{label}: {probabilities[idx]:.4f}")


# GPU-aware LIME Implementation
print("\n==== LIME Explanation ====")
explainer_lime = LimeTextExplainer(class_names=age_groups)

exp = explainer_lime.explain_instance(
    test_lyric,
    lambda x: predict(x, batch_size=8),  # Larger batch for LIME
    num_features=10,
    num_samples=500,
    labels=[predicted_class]  # Explicitly specify which class to explain
)
print(exp.available_labels())
exp.show_in_notebook(text=True)

print("Top influential words (LIME):")
for feature, weight in exp.as_list(label=predicted_class):
    print(f"{feature}: {weight:.4f}")


# SHAP Implementation
def f(word_list):
    all_output = []
    for text in word_list:
        # Handle both string and numpy array inputs
        if isinstance(text, np.ndarray):
            text = str(text)
            
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(model.device)
        mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(model.device)
        token_types = torch.tensor(encoding['token_type_ids']).unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            outputs = model(ids, mask, token_types)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_output.append(probs[0])  # Take first item since we process one at a time
    
    return np.array(all_output)

# Create SHAP explainer with proper tokenizer handling
explainer = shap.Explainer(
    f,
    masker=shap.maskers.Text(tokenizer=tokenizer, mask_token=tokenizer.mask_token),
    algorithm="partition",
    output_names=["semua usia", "anak", "remaja", "dewasa"]
)

shap_values = explainer([test_lyric])

print("\n==== SHAP Explanation ====")
shap.plots.text(shap_values[:, :, predicted_class])

shap.plots.text(shap_values)

shap.plots.bar(shap_values[0, :, predicted_class], max_display=10)

shap.plots.waterfall(shap_values[0, :, predicted_class], max_display=15)

# Get tokens (filtering out special tokens)
tokens = [token for token in tokenizer.tokenize(test_lyric) 
          if token not in ['[CLS]', '[SEP]', '[PAD]']]
valid_indices = [i for i, token in enumerate(tokenizer.tokenize(test_lyric)) 
                if token not in ['[CLS]', '[SEP]', '[PAD]']]
filtered_shap = shap_values[:, valid_indices, :]

# 4. Manual Decision Plot (since explainer.expected_value is None)
mean_prediction = probabilities.mean()  # Fallback base value
plt.figure(figsize=(12, 6))
shap.decision_plot(
    base_value=mean_prediction,
    shap_values=filtered_shap[0, :, predicted_class].values,
    features=tokens,
    feature_names=tokens,
    ignore_warnings=True,
    show=False
)
plt.title(f"Decision Process for '{age_groups[predicted_class]}' Prediction")
plt.tight_layout()
plt.show()