import re
from sklearn.model_selection import train_test_split

def process_text(text):
    text = re.sub(r'http\S+', '', text)     # Removes URLs
    text = re.sub(r'@\w+', '', text)        # Removes mentions
    text = re.sub(r'#\w+', '', text)        # Removes hashtags
    text = re.sub(r'&\w+;', '', text)       # Removes HTML entities
    text = re.sub(r'\d+', '', text)         # Removes numbers
    text = re.sub(r'[^\w\s]', '', text)     # Removes punctuation
    text = text.strip().lower()             # Converts to Lowercase and removes whitespace
    return text

def process_data(data):
    print("processing data...")
    data['text'] = data['text'].apply(process_text)
    return data 

def process_and_split(data):
    data=process_data(data)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    print("splitting data...")
    return train_test_split(texts, labels, test_size=0.2)