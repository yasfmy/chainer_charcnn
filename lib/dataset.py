import os
import pickle
import bz2
import xml.etree.ElementTree as ET
from operator import itemgetter, add
from functools import reduce
import random

import requests
import numpy as np

def fetch_ag_corpus(dest_path=None, shuffle_seed=0):
    xml_file = 'newsspace200.xml'
    if dest_path is None:
        dest_path = os.path.abspath('./dataset')
    else:
        dest_path = os.path.expanduser(dest_path)
    xml_path = os.path.join(dest_path, xml_file)
    if not os.path.isfile(xml_path):
        print('downloading a file...')
        r = requests.get('https://www.di.unipi.it/~gulli/newsspace200.xml.bz',
                          stream=True)
        compressed_path = xml_path + '.bz'
        with open(compressed_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print('expanding a file...')
        with bz2.open(compressed_path) as ins, open(xml_path, 'wb') as outs:
            outs.write(ins.read())

    pickle_file = 'newsspace200.pkl'
    pickle_path = os.path.join(dest_path, pickle_file)
    categories = {'World': 0, 'Entertainment': 1, 'Sports': 2, 'Business': 3}
    if not os.path.isfile(pickle_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        title = [el.text for el in root.findall('title')]
        desc = [el.text for el in root.findall('description')]
        category = [el.text for el in root.findall('category')]
        dataset = {'title': [], 'desc': [], 'label': []}
        for t, d, c in zip(title, desc, category):
            if c in categories:
                dataset['title'].append(t)
                dataset['desc'].append(d)
                dataset['label'].append(categories[c])
        with open(pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open(pickle_path, 'rb') as f:
            dataset = pickle.load(f)
        title = dataset['title']
        desc = dataset['desc']
    label = dataset['label']
    indices = list(range(len(label)))
    random.seed(shuffle_seed)
    random.shuffle(indices)
    title, desc, label = list(zip(*[(title[i], desc[i], label[i]) for i in indices]))
    splited = [sample_train_and_test(title, desc, label, c) for c in categories.values()]
    title_train, title_test, desc_train, desc_test, label_train, label_test \
        = [reduce(add, data) for data in list(zip(*splited))]
    return title_train, title_test, desc_train, desc_test, label_train, label_test

def sample_train_and_test(title, desc, label, category):
    getter = itemgetter(*np.where(np.array(label) == category)[0])
    title, desc, label = getter(title), getter(desc), getter(label)
    title_train, title_test = title[:30000], title[30000:31900]
    desc_train, desc_test = desc[:30000], desc[30000:31900]
    label_train, label_test = label[:30000], label[30000:31900]
    return title_train, title_test, desc_train, desc_test, label_train, label_test
