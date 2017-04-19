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
from mltools.trainer import ClassifierTrainer
from char_cnn import CharCNN
from dataset import AgCorpus
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
    y_train, x_train, y_test, x_test = AgCorpus(args.data_path).load_dataset()
    s2t = String2Tensor(char_table, char_table['unk'], args.length)
    x_train = [[s2t.encode(x)] for x in x_train]
    x_test = [[s2t.encode(x)] for x in x_test]

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
    order_iter = Sampling.provide_uniformly_sampled_order(
                            y_train, categories, batch_size)

    trainer = ClassifierTrainer(model, opt, x_train, y_train, x_test, y_test,
                ImageIterator, LabelIterator, batch_size, order_iter, n_epoch)
    trainer.run()

    model.save_model('model/charcnn.model', suffix=True)
    save_opt(opt, 'opt/charcnn.state', True)

if __name__ == '__main__':
    args = parse_args()
    main(args)
