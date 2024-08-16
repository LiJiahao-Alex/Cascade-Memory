dataset_info = {
    "MNIST":
        {
            "train_x": 'datasets/MNIST/preprocessed/train_data.npy',
            "train_y": 'datasets/MNIST/preprocessed/train_label.npy',
            "test_x": 'datasets/MNIST/preprocessed/test_data.npy',
            "test_y": 'datasets/MNIST/preprocessed/test_label.npy',
            "type": 'images',
            "shape": (1, 28, 28),
            "class_num": 10,
        },
}


def load_dataset(dataset_name):
    import numpy as np
    train_data = np.load(dataset_info[dataset_name]['train_x'], allow_pickle=True) if dataset_info[dataset_name]['train_x'] is not None else None
    train_label = np.load(dataset_info[dataset_name]['train_y'], allow_pickle=True) if dataset_info[dataset_name]['train_y'] is not None else None
    test_data = np.load(dataset_info[dataset_name]['test_x'], allow_pickle=True) if dataset_info[dataset_name]['test_x'] is not None else None
    test_label = np.load(dataset_info[dataset_name]['test_y'], allow_pickle=True) if dataset_info[dataset_name]['test_y'] is not None else None
    return train_data, train_label, test_data, test_label
