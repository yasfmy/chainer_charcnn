from math import ceil

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import Normal

from mltools.model import BaseModel

class CharCNN(BaseModel):
    def __init__(self, length, categories):
        fc_size = ceil((length - 96) / 27) * 256
        initializer = Normal(scale=0.05) # mean is 0, standard deviation is 0.05
        super().__init__(
                conv1 = L.Convolution2D(1, 256, (1, 7), initialW=initializer),
                conv2 = L.Convolution2D(256, 256, (1, 7), initialW=initializer),
                conv3 = L.Convolution2D(256, 256, (1, 3), initialW=initializer),
                conv4 = L.Convolution2D(256, 256, (1, 3), initialW=initializer),
                conv5 = L.Convolution2D(256, 256, (1, 3), initialW=initializer),
                conv6 = L.Convolution2D(256, 256, (1, 3), initialW=initializer),
                fc1 = L.Linear(fc_size, 1024),
                fc2 = L.Linear(1024, 1024),
                fc3 = L.Linear(1024, categories)
        )

    def loss(self, x, t):
        y = self.__forward(x, train=True)
        return F.softmax_cross_entropy(y, t)

    def accuracy(self, x, t):
        y = self.__forward(x)
        return F.accuracy(y, t)

    def __forward(self, x, train=False):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, (1, 3), stride=3)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, (1, 3), stride=3)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(h, (1, 3), stride=3)
        h = F.dropout(F.relu(self.fc1(h)), ratio=0.5, train=train)
        h = F.dropout(F.relu(self.fc2(h)), ratio=0.5, train=train)
        y = self.fc3(h)
        return y
