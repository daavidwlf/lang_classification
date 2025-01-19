import re
from sklearn.model_selection import train_test_split

def process_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'&\w+;', '', text)  # Remove HTML entities
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip().lower()  # Lowercase and strip whitespace  # Remove extra spaces
    return text

def process_data(data):
    print("processing data...")
    data['text'] = data['text'].apply(process_text)
    return data 

def process_and_split(data):
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    print("splitting data...")
    return train_test_split(texts, labels, test_size=0.2)
