import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import processing

def load_model_and_tokenizer(saved_model_path):
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
    return tokenizer, model

def choose_device(model):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    return device

def predict_text(text, tokenizer, model, device, max_len=64):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id

def predict(X_test):
    saved_model_path = "../saved_model"
    tokenizer, model = load_model_and_tokenizer(saved_model_path)
    device = choose_device(model)
    
    predictions = []
    for text in X_test["text"]:
        pred = predict_text(text, tokenizer, model, device)
        predictions.append(pred)
    
    return predictions

data_test = pd.read_csv("../data/test.csv")

X_test = processing.process_data(data_test)

y_test = predict(X_test)

df = pd.DataFrame({"id": X_test["id"], "text": X_test["text"], "label": y_test})
df.to_csv("../results/test_with_label.csv", index=False)
print("Prediction complete. Results saved to ../results/test_with_label.csv")