import pandas as pd
import re
import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('--train_file', type=str, help='Input dir for videos')
    parser.add_argument('--test_file', type=str, help='Input dir for videos')
    args = parser.parse_args()


    train = pd.read_excel(args.train_file)
    print(f"Training file loaded: {args.train_file}")

    test = pd.read_excel(args.test_file)
    print(f"Testing file loaded: {args.test_file}")

    print(f"Length of training dataset is {len(train)}")
    print(f"Length of testing dataset is {len(test)}")

    train['input'] = train['input'].str.replace('\n', '')
    train['input'] = train['input'].str.replace(':', '')
    train['input'] = train['input'].str.replace(';', '')

    train['input'] = train['input'].apply(lambda row: re.sub(r"[^a-zA-Z0-9]", " ", str(row)))

    train["len"] = train.apply(lambda row: len(row["input"]), axis=1)

    train = train[train["len"] > 20]

    train = train.dropna()

    test['input'] = test['input'].str.replace('\n', '')
    test['input'] = test['input'].str.replace(':', '')
    test['input'] = test['input'].str.replace(';', '')

    test['input'] = test['input'].apply(lambda row: re.sub(r"[^a-zA-Z0-9]", " ", str(row)))

    test["len"] = test.apply(lambda row: len(row["input"]), axis=1)

    test = test[test["len"] > 20]


    print("Text preprocessing is done")

    train_texts = train["input"].to_list()

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(train["label"])

    train["label"] = le.transform(train["label"])
    train_labels = train["label"].to_list()

    test_texts = test["input"].to_list()
    test["label"] = le.transform(test["label"])
    test_labels = test["label"].to_list()

    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    import torch

    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=3,  # batch size per device during training
        per_device_eval_batch_size=3,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()

    prediction = trainer.predict(test_dataset)

    import numpy as np
    unique, counts = np.unique(prediction.label_ids, return_counts=True)
    dict(zip(unique, counts))

    unique, counts = np.unique(np.array(test_labels), return_counts=True)
    dict(zip(unique, counts))

    from sklearn.metrics import accuracy_score
    print("Accuracy is ",accuracy_score(prediction.label_ids, test_labels)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
