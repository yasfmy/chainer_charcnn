from chainer import serializers

def save_opt(opt, filename):
    serializers.save_npz(filename, opt)

def load_opt(opt, filename):
    serializers.load_npz(filename, opt)
