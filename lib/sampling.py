import numpy as np

def uniformly_sampling(dataset_size, sample_size, categories):
    each_size = dataset_size // categories
    each_sample_size = sample_size // categories
    indices = []
    for _ in range(categories):
        indices.append(np.random.permutation(each_size) + each_size * categories)
    order = np.array([])
    for i in range(0, each_size, each_sample_size):
        i_end = i + each_sample_size
        for j in range(categories):
            order = np.concatenate((order, indices[j][i:i_end]))
    return order.astype(np.int32)
