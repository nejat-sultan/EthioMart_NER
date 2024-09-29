import os
import pandas as pd  # type: ignore
import re

def preprocess_text(text):
    if isinstance(text, str):  
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^ሀ-ፐA-Za-z0-9\s]', '', text)  # Keep only certain characters
        text = ' '.join(text.split())  # Remove extra spaces
        return text
    else:
        return '' 

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    df['text_cleaned'] = df['text'].apply(preprocess_text)

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    df.to_csv(output_file, index=False)

preprocess_data('raw_data/messages.csv', 'data/cleaned_messages.csv')
