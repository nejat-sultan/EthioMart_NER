import pandas as pd  # type: ignore
import os

def label_messages(messages):
    labeled_data = []

    for message in messages:
        text = message['text']
        
        if isinstance(text, str):
            tokens = text.split()
        else:
            continue  

        for token in tokens:
            label = 'O'  

            if token in ["ዋጋ", "1000", "ብር"]: 
                if label == 'O':
                    label = 'B-PRICE'
                else:
                    label = 'I-PRICE'
            elif token in ["Addis", "abeba", "Bole"]: 
                if label == 'O':
                    label = 'B-LOC'
                else:
                    label = 'I-LOC'
            elif token in ["Baby", "bottle"]: 
                if label == 'O':
                    label = 'B-Product'
                else:
                    label = 'I-Product'

            labeled_data.append((token, label))
        
        labeled_data.append(("", ""))  

    return labeled_data

def save_to_conll(labeled_data, filename='labeled_data.conll'):
    with open(filename, 'w', encoding='utf-8') as f:
        for token, label in labeled_data:
            f.write(f"{token} {label}\n")
        f.write("\n")  

def main():
    messages_df = pd.read_csv('raw_data/messages.csv')

    labeled_data = label_messages(messages_df.to_dict(orient='records'))

    save_to_conll(labeled_data)

if __name__ == "__main__":
    main()
