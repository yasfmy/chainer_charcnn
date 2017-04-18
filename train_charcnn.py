import os
import math
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer import optimizers
from chainer.optimizer import WeightDecay

from mltools.preprocessing import String2Tensor, char_table
from mltools.iterator import ImageIterator, LabelIterator
from mltools.sampling import Sampling
from char_cnn import CharCNN
from dataset import AgCorpus
from sampling import uniformly_sampling
from optimizer import save_opt, load_opt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--categories', default=4, type=int)
    parser.add_argument('--length', default=1024, type=int)
    parser.add_argument('--data-path', type=str,
            default=os.path.abspath('./dataset'))
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--decay', default=1e-5, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--opt', default=None, type=str)
    return parser.parse_args()

def main(args):
    train_label, train_data, test_label, test_data = \
                                    AgCorpus(args.data_path).load_dataset()
    s2t = String2Tensor(char_table, char_table['unk'], args.length)
    train_data = [[s2t.encode(x)] for x in train_data]
    test_data = [[s2t.encode(x)] for x in test_data]

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
    N = len(train_label)
    N_test = len(test_label)
    order_iter = Sampling.provide_uniformly_sampled_order(
                            train_label, categories, batch_size)

    for i in range(n_epoch):
        epoch = i + 1
        print('epoch: {}'.format(epoch))
        order = next(order_iter)
        train_iter = ImageIterator(train_data, batch_size, order=order, gpu=gpu_flag)
        label_iter = LabelIterator(train_label, batch_size, order=order, gpu=gpu_flag)
        sum_loss = 0
        if epoch % 3 == 0 and epoch <= 30:
            opt.lr /= 2
        for X, y in zip(train_iter, label_iter):
            model.cleargrads()
            loss = model.loss(X, y)
            loss.backward()
            opt.update()
            sum_loss += loss.data * len(y)
        print('loss: {}'.format(sum_loss / N))

        test_iter = ImageIterator(test_data, batch_size, shuffle=False, gpu=gpu_flag)
        label_iter = LabelIterator(test_label, batch_size, shuffle=False, gpu=gpu_flag)
        sum_accuracy = 0
        for X, y in zip(test_iter, label_iter):
            accuracy = model.accuracy(X, y)
            sum_accuracy += accuracy.data * len(y)
        print('accuracy: {}'.format(sum_accuracy / N_test))

    model.save_model('model/charcnn.model', True)
    save_opt(opt, 'opt/charcnn.state', True)

if __name__ == '__main__':
    args = parse_args()
    main(args)
