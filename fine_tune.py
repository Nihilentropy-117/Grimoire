from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
import numpy as np
import evaluate
import torch
from semantic_split import SimilarSentenceSplitter, SentenceTransformersSimilarity, Splitter
import os
import pickle
import spacy


class SpacySentenceSplitter(Splitter):

    def __init__(self):
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
        nlp.max_length = 30000000
        nlp.add_pipe('sentencizer')
        self.nlp = nlp

    def split(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [str(sent).strip() for sent in doc.sents]


if torch.cuda.is_available():
    print("GPU is available. Current device:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU is not available, using CPU instead.")


def make_dataset(_dataset, _type, _model, group_max_sentences=5):
    tokenizer = AutoTokenizer.from_pretrained(_model)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    pickle_file = 'dataset.pickle'

    if _type == "datasets":
        dataset = load_dataset(_dataset)
    elif _type == "raw":
        if os.path.exists(pickle_file):
            # Load the previously saved dataset from pickle file
            with open(pickle_file, 'rb') as f:
                print("Loading dataset from pickle file.")
                dataset = pickle.load(f)
        else:
            # Process the dataset since no pickle file exists
            with open(_dataset, 'r', encoding='utf-8') as f:
                text = f.read()
                model = SentenceTransformersSimilarity()
                sentence_splitter = SpacySentenceSplitter()
                splitter = SimilarSentenceSplitter(model, sentence_splitter)
                # Split the text using the SimilarSentenceSplitter
                chunks = splitter.split(text, group_max_sentences=group_max_sentences)
                overlapping_chunks = []
                for i in range(len(chunks) - 3):
                    # Collect sentences for the overlap before, current chunk, and overlap after
                    overlap_before = sum(chunks[max(0, i - 1):i], [])
                    current_chunk = sum(chunks[i:i + 3], [])
                    overlap_after = sum(chunks[i + 3:i + 4], [])

                    # Combine into a single chunk
                    full_chunk = overlap_before + current_chunk + overlap_after
                    overlapping_chunks.append(full_chunk)

                dataset = Dataset.from_dict({"text": overlapping_chunks})

                # Save the processed dataset
                with open(pickle_file, 'wb') as f:
                    print("Saving dataset to pickle file.")
                    pickle.dump(dataset, f)
    else:
        print("Invalid Dataset Type")
        return None

    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True), batched=True)

    total_entries = len(tokenized_dataset)
    split_index = int(total_entries * 0.95)  # Calculate the 95%split

    train_dataset = tokenized_dataset[:split_index]  # 95% of the datase
    eval_dataset = tokenized_dataset[split_index:]  # Remaining 5% 

    return train_dataset, eval_dataset


def train(model, datasets):
    train_dataset, eval_dataset = datasets

    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=5)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(

        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,

    )

    trainer.train()


if __name__ == "__main__":
    model = "mosaicml/mpt-7b-storywriter"

    """
    ds = "yelp_review_full"
    train, eval = make_dataset(ds, "datasets", model, chunk_size=2048)
    """

    ds = "raw_trainer.txt"
    training, evaluation = make_dataset(ds, "raw", model, group_max_sentences=5)

    if training:
        train(model, [training, evaluation])
