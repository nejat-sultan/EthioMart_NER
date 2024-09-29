import pandas as pd  # type: ignore
import os
import torch  # type: ignore
from transformers import (  # type: ignore
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import precision_recall_fscore_support  
from datasets import load_dataset  # type: ignore
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score  # type: ignore


class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_conll(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence, current_labels = [], []
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split()
                current_sentence.append(token)
                current_labels.append(label)
            else:
                if current_sentence:  
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                current_sentence, current_labels = [], []
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    return sentences, labels


def tokenize_and_align_labels(sentences, labels, tokenizer):
    tokenized_inputs = []
    aligned_labels = []
    for sentence, label in zip(sentences, labels):
        encoding = tokenizer(sentence, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        input_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']

        aligned_label = []
        current_label_index = 0

        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                aligned_label.append(-100)  
            elif current_label_index < len(label) and i == offsets[current_label_index][0]:
                aligned_label.append(label[current_label_index])
                current_label_index += 1
            else:
                aligned_label.append(-100)

        tokenized_inputs.append(input_ids)
        aligned_labels.append(aligned_label)

    max_length = max(len(ids) for ids in tokenized_inputs)
    padded_inputs = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in tokenized_inputs]
    padded_labels = [lbl + [-100] * (max_length - len(lbl)) for lbl in aligned_labels]

    return padded_inputs, padded_labels


def fine_tune_model(train_file, val_file):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("C:/Users/nejat/AIM Projects/EthioMart_NER/xlm-roberta-base")
    model = XLMRobertaForTokenClassification.from_pretrained("C:/Users/nejat/AIM Projects/EthioMart_NER/xlm-roberta-base", num_labels=5)

    train_sentences, train_labels = load_conll(train_file)
    train_tokenized_inputs, train_aligned_labels = tokenize_and_align_labels(train_sentences, train_labels, tokenizer)

    val_sentences, val_labels = load_conll(val_file)
    val_tokenized_inputs, val_aligned_labels = tokenize_and_align_labels(val_sentences, val_labels, tokenizer)

    train_dataset = TokenClassificationDataset(train_tokenized_inputs, train_aligned_labels)
    val_dataset = TokenClassificationDataset(val_tokenized_inputs, val_aligned_labels)

    training_args = TrainingArguments(
        output_dir='./models/fine_tuned_model',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained('./models/fine_tuned_model')
    tokenizer.save_pretrained('./models/fine_tuned_model')


def compute_metrics(prediction: EvalPrediction):
    true_labels = prediction.label_ids
    true_predictions = prediction.predictions.argmax(-1)

    true_labels_flat = [label for label in true_labels.flatten() if label != -100]
    true_predictions_flat = [pred for label, pred in zip(true_labels.flatten(), true_predictions.flatten()) if label != -100]

    if len(true_labels_flat) == 0 or len(true_predictions_flat) == 0:
        return {"accuracy": 0.0}

    accuracy = accuracy_score(true_labels_flat, true_predictions_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, true_predictions_flat, average='weighted')

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":
    fine_tune_model('labeled_data.conll', 'labeled_data.conll') 