from torchvision.datasets import mnist
import numpy as np

train_data = mnist.read_image_file('train-images.idx3-ubyte').numpy()
train_label = mnist.read_label_file('train-labels.idx1-ubyte').numpy()
test_data = mnist.read_image_file('t10k-images.idx3-ubyte').numpy()
test_label = mnist.read_label_file('t10k-labels.idx1-ubyte').numpy()

np.save('preprocessed/train_data.npy', np.expand_dims(train_data, 1))
np.save('preprocessed/train_label.npy', train_label)
np.save('preprocessed/test_data.npy', np.expand_dims(test_data, 1))
np.save('preprocessed/test_label.npy', test_label)
