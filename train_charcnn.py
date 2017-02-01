import os
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer import optimizers
from tools.text.preprocessing import OneOfMEncoder, char_table
from tools.iterator import ImageIterator, LabelIterator

from lib.char_cnn import CharCNN
from lib.dataset import fetch_ag_corpus
from lib.sampling import uniformly_sampling
from lib.optimizer import save_opt, load_opt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--categories', default=4, type=int)
    parser.add_argument('--length', default=1024, type=int)
    parser.add_argument('--data_file', type=str,
            default=os.path.abspath('./dataset'))
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--seed', default=123456, type=int)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--opt', default=None, type=str)
    return parser.parse_args()

def main(args):
    title_train, title_test, desc_train, desc_test, label_train, label_test \
        = fetch_ag_corpus(args.data_file, args.seed)
    one_of_m = OneOfMEncoder(char_table, char_table['unk'], args.length)
    text_train = [[one_of_m.encode('{}\n{}'.format(t, d))]
                    for t, d in zip(title_train, desc_train)]
    text_test = [[one_of_m.encode('{}\n{}'.format(t, d))]
                    for t, d in zip(title_test, desc_test)]

    categories = args.categories
    model = CharCNN(args.length, categories)
    if args.model:
        model.load_model(args.model)
    gpu_id = args.gpu
    gpu_flag = True if gpu_id >= 0 else False
    if gpu_flag:
        model.use_gpu(gpu_id)
    opt = optimizers.MomentumSGD(args.lr, args.momentum)
    opt.setup(model)
    if args.opt:
       load_opt(opt, args.opt)

    batch_size = args.batch
    n_epoch = args.epoch
    N = len(label_train)
    N_test = len(label_test)

    for i in range(n_epoch):
        epoch = i + 1
        print('epoch: {}'.format(epoch))
        order = uniformly_sampling(N, batch_size, categories)
        text_iter = ImageIterator(text_train, batch_size, order=order, gpu=gpu_flag)
        label_iter = LabelIterator(label_train, batch_size, order=order, gpu=gpu_flag)
        sum_loss = 0
        if epoch % 3 == 0 and epoch <= 30:
            opt.lr /= 2
        for X, y in zip(text_iter, label_iter):
            model.cleargrads()
            loss = model.loss(X, y)
            loss.backward()
            opt.update()
            sum_loss += loss.data * len(y)
        print('loss: {}'.format(sum_loss / N))

        text_iter = ImageIterator(text_test, batch_size, shuffle=False, gpu=gpu_flag)
        label_iter = LabelIterator(label_test, batch_size, shuffle=False, gpu=gpu_flag)
        sum_accuracy = 0
        for X, y in zip(text_iter, label_iter):
            accuracy = model.accuracy(X, y)
            sum_accuracy += accuracy.data * len(y)
        print('accuracy: {}'.format(sum_accuracy / N_test))

    model.save_model('model/charcnn.model', True)
    save_opt(opt, 'opt/charcnn.state', True)

if __name__ == '__main__':
    args = parse_args()
    main(args)
