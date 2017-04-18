import os
import pickle
import bz2
import tarfile
import csv
import random

import requests
import numpy as np

class AgCorpus:

    download_link = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms'

    def __init__(self, save_path):
        self.save_path = save_path
        self.tar_file_path = os.path.join(save_path, 'ag_news_csv.tar.gz')
        self.data_path = os.path.join(save_path, 'ag_news_csv')
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.test_path = os.path.join(self.data_path, 'test.csv')

    def download(self):
        dest_path = self.tar_file_path
        if os.path.isfile(dest_path):
            return
        r = requests.get(self.download_link, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def expand(self):
        if os.path.isdir(self.data_path):
            return
        f = tarfile.open(self.tar_file_path)
        f.extractall(self.save_path)
        f.close()

    def prepare(self):
        self.download()
        self.expand()

    def load_dataset(self):
        self.prepare()
        with open(self.train_path) as f:
            train_label, train_data = list(zip(*(
                (int(row[0]) - 1, '{} {}'.format(row[1], row[2]).lower())
                for row in csv.reader(f))))
        with open(self.test_path) as f:
            test_label, test_data = list(zip(*(
                (int(row[0]) - 1, '{} {}'.format(row[1], row[2]).lower())
                for row in csv.reader(f))))
        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data
        return train_label, train_data, test_label, test_data
