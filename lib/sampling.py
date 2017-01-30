import numpy as np

def uniformly_sampling(sample_size, label, categories):
    dataset_size = len(label)
    each_size = dataset_size // categories
    each_sample_size = sample_size // categories
    indices = []
    for _ in range(categories):
        indices.append(np.random.permutation(each_size) + each_size * categories)
    order = np.array([])
    for i in range(0, each_sample_size, each_size):
        i_end = i + each_sample_size
        for j in range(categories):
            order = np.concatenate((order, categories[j][i:i_end]))
    return order
