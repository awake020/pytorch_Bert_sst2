import transformers
from transformers import *
import csv
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np


class DataIOSST2(object):
    def __init__(self, config):
        self.path = config['path']
        self.batch_size = config['batch_size']
        self.train_word, self.train_label, \
        self.dev_word, self.dev_label, \
        self.test_word, \
        self.test_label = self.read_train_dev_test()

    def read_train_dev_test(self):
        train_word, train_label = self.read_data(self.path + '/train.tsv')
        dev_word, dev_label = self.read_data(self.path + '/dev.tsv')
        test_word, test_label = self.read_data(self.path + '/test.tsv')
        return train_word, train_label, dev_word, dev_label, test_word, test_label

    @staticmethod
    def read_data(path):
        data = []
        label = []
        csv.register_dialect('my', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(path) as tsvfile:
            file_list = csv.reader(tsvfile, "my")
            first = True
            for line in file_list:
                if first:
                    first = False
                    continue
                data.append(line[1])
                label.append(int(line[0]))
        csv.unregister_dialect('my')
        return data, label


class SSTDataset(Dataset):
    def __init__(self, sentences, labels):
        self.dataset = sentences
        self.labels = labels

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)


dataset = DataIOSST2({'path': '/SST-2', 'batch_size': 16})
train_set = SSTDataset(dataset.train_word, dataset.train_label)
train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
train_eval_loader = DataLoader(train_set, shuffle=False, batch_size=16)

dev_set = SSTDataset(dataset.dev_word, dataset.dev_label)
dev_eval_loader = DataLoader(dev_set, shuffle=False, batch_size=16)

test_set = SSTDataset(dataset.test_word, dataset.test_label)
test_eval_loader = DataLoader(test_set, shuffle=False, batch_size=16)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, hidden_dropout_prob=0.3)
model.cuda(device=0)
optimizer = transformers.AdamW(model.parameters(), lr=2e-5)


def train():
    loss = 0.0
    print('epoch', epoch + 1)
    model.train()
    for x, y in tqdm(train_loader):
        inputs = tokenizer(x, return_tensors='pt', padding=True)
        labels = y
        for w in inputs:
            inputs[w] = inputs[w].cuda(device=0)
        labels = labels.cuda(device=0)
        out = model(**inputs, labels=labels)
        out[0].backward()
        optimizer.step()
        loss += out[0].item()
        optimizer.zero_grad()
    print(loss / len(train_loader))


def test(loader, data_set_name):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for x, y in tqdm(loader):
            inputs = tokenizer(x, return_tensors='pt', padding=True)
            labels = y
            for w in inputs:
                inputs[w] = inputs[w].cuda(device=0)
            labels = labels.cuda(device=0)
            out = model(**inputs)
            pred = torch.argmax(out, dim=-1, keepdim=False)
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()
            correct += np.sum(pred == labels)
            total += pred.shape[0]
        print(data_set_name, correct / total)


for epoch in range(20):
    train()
    test(train_eval_loader, "train")
    test(test_eval_loader, "test")
    test(dev_eval_loader, 'dev')
