import pandas as pd
import re
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
import numpy as np

from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPTNeoForSequenceClassification

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('--train_file', type=str, help='Input dir for videos')
    parser.add_argument('--test_file', type=str, help='Input dir for videos')
    parser.add_argument('--valid_file', type=str, help='Input dir for videos')
    parser.add_argument('--batch_size', type=int, help='Input dir for videos')
    args = parser.parse_args()

    train = pd.read_excel(args.train_file)
    print(f"Training file loaded: {args.train_file}")
    print(train.head())

    valid = pd.read_excel(args.valid_file)
    print(f"Training file loaded: {args.valid_file}")
    print(valid.head())

    test = pd.read_excel(args.test_file)
    print(f"Testing file loaded: {args.test_file}")
    print(test.head())


    print(f"Length of training dataset is {len(train)}")
    print(f"Length of training dataset is {len(valid)}")
    print(f"Length of testing dataset is {len(test)}")


    train['input'] = train['input'].str.replace('\n', '')
    train['input'] = train['input'].str.replace(':', '')
    train['input'] = train['input'].str.replace(';', '')
    train['input'] = train['input'].apply(lambda row: re.sub(r"[^a-zA-Z0-9]", " ", str(row)))
    train["len"] = train.apply(lambda row: len(row["input"]), axis=1)
    train = train[train["len"] > 20]
    train = train.dropna()


    valid['input'] = valid['input'].str.replace('\n', '')
    valid['input'] = valid['input'].str.replace(':', '')
    valid['input'] = valid['input'].str.replace(';', '')
    valid['input'] = valid['input'].apply(lambda row: re.sub(r"[^a-zA-Z0-9]", " ", str(row)))
    valid["len"] = valid.apply(lambda row: len(row["input"]), axis=1)
    valid = valid[valid["len"] > 20]
    valid = valid.dropna()


    test['input'] = test['input'].str.replace('\n', '')
    test['input'] = test['input'].str.replace(':', '')
    test['input'] = test['input'].str.replace(';', '')
    test['input'] = test['input'].apply(lambda row: re.sub(r"[^a-zA-Z0-9]", " ", str(row)))
    test["len"] = test.apply(lambda row: len(row["input"]), axis=1)
    test = test[test["len"] > 20]
    train = train.dropna()


    print(f"Length of training dataset after preprocessing is {len(train)}")
    print(f"Length of testing dataset after preprocessing is {len(test)}")

    print("Text preprocessing is done")

    train_texts = train["input"].to_list()

    le = preprocessing.LabelEncoder()
    le.fit(train["label"])

    import numpy
    numpy.save("classes.npy", le.classes_)

    print("Classes are saved")

    #
    # train["label"] = le.transform(train["label"])
    train_labels = train["label"].to_list()

    print(f"Train texts ", len(train_texts))
    print(f"Train labels ", len(train_labels))

    val_texts = valid["input"].to_list()
    # valid["label"] = le.transform(valid["label"])
    val_labels = valid["label"].to_list()

    print(f"Val texts ", len(val_texts))
    print(f"Val labels ", len(val_labels))

    test_texts = test["input"].to_list()
    # test["label"] = le.transform(test["label"])
    test_labels = test["label"].to_list()

    print(f"Test texts ", len(test_texts))
    print(f"Test labels ", len(test_labels))

    print(f"Number of label is {len(train['label'].unique())}")
    #
    # train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    #
    # dictionary of lists
    # dict = {'input': train_texts, "label": le.inverse_transform(train_labels)}
    # df = pd.DataFrame(dict)
    # df.to_csv("train.csv", sep=";")
    #
    # dict = {'input': val_texts, "label": le.inverse_transform(val_labels)}
    # df = pd.DataFrame(dict)
    # df.to_csv("val.csv", sep=";")
    #
    # dict = {'input': test_texts, "label": le.inverse_transform(test_labels)}
    # df = pd.DataFrame(dict)
    # df.to_csv("test.csv", sep=";")

    #
    # del(df)

    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    # tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    # tokenizer.pad_token = tokenizer.eos_token

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    print("Encoding ended")


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

    print("Datasets prepared")

    training_args = TrainingArguments(
        output_dir='./New_multi',  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs_new_multi_more_epochs',  # directory for storing logs
        save_steps=5000,
        logging_steps=100,
    )

    # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
    #                                                             num_labels=len(train['label'].unique()),
    #                                                             ignore_mismatched_sizes=True)
    # model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli",
    #                                                          num_labels=len(train['label'].unique()),
    #                                                          ignore_mismatched_sizes=True)

    # model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment",
    #                                                          num_labels=len(train['label'].unique()),
    #                                                          ignore_mismatched_sizes=True)


    model = BertForSequenceClassification.from_pretrained("zyro/ai-what-is-web/checkpoint-50000/",
                                                             num_labels=len(train['label'].unique()),
                                                             ignore_mismatched_sizes=True)


    # model = GPTNeoForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-125M',
    #                                                          num_labels=len(train['label'].unique()),
    #                                                          ignore_mismatched_sizes=True)

    trainer = Trainer(
        model=model,  # the instantiated ???? Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report

    # # Evaluation on training dataset
    prediction = trainer.predict(train_dataset)
    preds = np.argmax(prediction.predictions, axis=-1)
    print(accuracy_score(preds, prediction.label_ids))
    # target_names = le.inverse_transform(list(range(0, len(train["label"].unique()))))
    print(classification_report(preds,
                                prediction.label_ids,
                                # target_names=target_names,
                                digits=3))

    pred_labels = preds
    with open('train_predictions.txt', 'w') as f:
        for item in pred_labels:
            f.write("%s\n" % item)



    # # Evaluation on validation dataset
    prediction = trainer.predict(val_dataset)
    preds = np.argmax(prediction.predictions, axis=-1)
    print(accuracy_score(preds, prediction.label_ids))
    # target_names = le.inverse_transform(list(range(0, len(train["label"].unique()))))
    print(classification_report(preds,
                                prediction.label_ids,
                                # target_names=target_names,
                                digits=3))

    pred_labels = preds
    with open('valid_predictions.txt', 'w') as f:
        for item in pred_labels:
            f.write("%s\n" % item)




    # Evaluation on testing dataset
    prediction = trainer.predict(test_dataset)
    preds = np.argmax(prediction.predictions, axis=-1)
    print(accuracy_score(preds, prediction.label_ids))
    # target_names = le.inverse_transform(list(range(0, len(train["label"].unique()))))
    print(classification_report(preds,
                                prediction.label_ids,
                                # target_names=target_names,
                                digits=3))

    pred_labels = preds
    with open('test_predictions.txt', 'w') as f:
        for item in pred_labels:
            f.write("%s\n" % item)



if __name__ == '__main__':
    main()
