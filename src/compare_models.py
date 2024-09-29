import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast

class NERDataset(Dataset):
    def __init__(self, tokenized_inputs, aligned_labels):
        self.tokenized_inputs = tokenized_inputs
        self.aligned_labels = aligned_labels

    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.tokenized_inputs['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.tokenized_inputs['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.aligned_labels[idx], dtype=torch.long),  
        }
        return item

def load_conll(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence = []
        current_labels = []
        
        for line in file:
            if line.strip():  
                word, label = line.split()  
                current_sentence.append(word)
                current_labels.append(label)
            else:
                if current_sentence:  
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
        
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
    return sentences, labels

def tokenize_and_align_labels(sentences, labels, tokenizer):
    label_list = list(set(label for sublist in labels for label in sublist))
    label_map = {label: idx for idx, label in enumerate(label_list)}
    label_map['O'] = 0  
    
    tokenized_inputs = tokenizer(sentences, 
                                  is_split_into_words=True, 
                                  padding=True, 
                                  truncation=True, 
                                  return_tensors="pt")

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        aligned_label = [-100] * len(tokenized_inputs['input_ids'][i])  

        for word_index in range(len(label)):
            if word_ids[word_index] is not None:
                aligned_label[word_ids[word_index]] = label_map[label[word_index]] 

        aligned_labels.append(aligned_label)

    return tokenized_inputs, aligned_labels

def fine_tune_model(model_name):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)


    sentences, labels = load_conll('labeled_data.conll')
    print(f"Loaded {len(sentences)} sentences with {sum(len(lbl) for lbl in labels)} labels.")

    model = XLMRobertaForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(set(label for sublist in labels for label in sublist)) + 1  # +1 for 'O' label
    )

    tokenized_inputs, aligned_labels = tokenize_and_align_labels(sentences, labels, tokenizer)

    train_dataset = NERDataset(tokenized_inputs, aligned_labels)

    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    total_steps = len(train_dataset) // 16 * 3  

    training_args = TrainingArguments(
        output_dir=f'./models/{model_name.replace("/", "_")}_fine_tuned',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        eval_steps=100, 
        evaluation_strategy="steps", 
        max_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
    )

    trainer.train()
    return trainer.evaluate()



def compare_models():
    models = [
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased"
    ]
    
    results = {}
    for model_name in models:
        print(f"Fine-tuning {model_name}...")
        eval_results = fine_tune_model(model_name)
        results[model_name] = eval_results
        print(f"{model_name} evaluation results: {eval_results}")

    best_model = max(results, key=lambda k: results[k]['eval_accuracy'])
    print(f"Best performing model: {best_model}")

if __name__ == "__main__":
    compare_models()
