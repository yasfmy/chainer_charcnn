import os
from datetime import datetime

from chainer import serializers

def save_opt(opt, filename, suffix=False):
    if suffix:
        now = datetime.now().strftime('%Y%m%d')
        root, ext = os.path.splitext(filename)
        filename = '{}{}{}'.format(root, now, ext)
    serializers.save_npz(filename, opt)

def load_opt(opt, filename):
    serializers.load_npz(filename, opt)
