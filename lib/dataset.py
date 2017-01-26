import os
from subprocess import check_output, Popen
import xml.etree.ElementTree as ET

import requests

def fetch_ag_corpus(dest_path=None):
    if dest_path is None:
        dest_path = os.path.abspath('./dataset/newsspace200.xml')
    else:
        dest_path = os.path.expanduser(dest_path)
    if not os.path.isfile(dest_path):
        print('downloading a file...')
        r = requests.get('https://www.di.unipi.it/~gulli/newsspace200.xml.bz',
                          stream=True)
        compressed_file = dest_path + '.bz'
        with open(compressed_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print('expanding a file...')
        expand_command = check_output(['which', 'bunzip2']).decode('utf-8').strip()
        command = [expand_command, '-c', compressed_file]
        with open(dest_path, 'wb') as outs:
            process = Popen(command, stdout=outs)
            process.wait()

    tree = ET.parse(dest_path)
    root = tree.getroot()
    categories = {'World': 0, 'Entertainment': 1, 'Sports': 2, 'Business': 3}
    title = [el.text for el in root.findall('title')]
    description = [el.text for el in root.findall('description')]
    category = [el.text for el in root.findall('category')]
    dataset = {'title': [], 'desc': [], 'label': []}
    for t, d, c in zip(title, description, category):
        if c in categories:
            dataset['title'].append(t)
            dataset['desc'].append(d)
            dataset['label'].append(categories[c])
    return dataset
