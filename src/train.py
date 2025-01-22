import pandas as pd
import processing
from transformers import BertTokenizer
import model
import numpy as np
import torch
import random

seed=42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data = pd.read_csv("../data/train.csv")

X_train, X_val, y_train, y_val = processing.process_and_split(data)

model_name = 'bert-base-uncased'
max_len = 32
num_labels = 3
num_epochs = 3

model.train_and_save(X_train, y_train, X_val, y_val, model_name, num_labels, max_len, num_epochs)

print("Done")