import torch
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tool.dataset_map import load_dataset, dataset_info
from tool.plugin import ZScore_norm, MinMax_norm


def train(model,param):
    train_data, train_label, test_data, test_label = load_dataset(param.DATASET)
    train_data = train_data[train_label != param.ANOMALY_ID]
    train_label = train_label[train_label != param.ANOMALY_ID]
    test_label = (test_label == param.ANOMALY_ID)
    trans = [MinMax_norm, ZScore_norm]
    if dataset_info[param.DATASET]["type"] == "images":
        for f in trans:
            train_data = f(train_data)
            test_data = f(test_data)
    trainData, valData, trainLabel, valLabel = train_test_split(train_data,
                                                                train_label,
                                                                test_size=param.VAL_SIZE,
                                                                shuffle=param.SPILT_SHUFFLE,
                                                                random_state=param.SPILT_SEED
                                                                )
    train_dataset = TensorDataset(torch.from_numpy(trainData), torch.from_numpy(trainLabel))
    val_dataset = TensorDataset(torch.from_numpy(valData), torch.from_numpy(valLabel))
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=param.TRAIN_BATCH_SIZE,
                              shuffle=param.LOADER_SHUFFLE,
                              num_workers=param.NUM_WORKERS,
                              pin_memory=param.PIN_MEMORY,
                              drop_last=param.DROP_LAST,
                              )
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=param.VAL_BATCH_SIZE,
                            shuffle=param.LOADER_SHUFFLE,
                            num_workers=param.NUM_WORKERS,
                            pin_memory=param.PIN_MEMORY,
                            drop_last=False,
                            )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=param.BATCH_SIZE,
                             shuffle=param.LOADER_SHUFFLE,
                             num_workers=param.NUM_WORKERS,
                             pin_memory=param.PIN_MEMORY,
                             drop_last=False
                             )

    x = PrettyTable()
    x.field_names = ["train_data", "train_label", "test_data", "test_label"]
    x.add_row([train_data.shape, train_label.shape, test_data.shape, test_label.shape])
    print(x)

    x = PrettyTable()
    x.field_names = ["#positive(1)", "#negative(0)", "#total", "test-ρ"]
    pos = len(test_label[test_label == 1])
    neg = len(test_label[test_label == 0])
    total = len(test_label)
    x.add_row([pos, neg, total, round(pos / total, 4)])
    print(x)

    x = PrettyTable()
    x.field_names = ["train_data", "train_label", "-->", "trainData", "trainLabel", "valData", "valLabel"]
    x.add_row(
        [train_data.shape, train_label.shape, "-->", trainData.shape, trainLabel.shape, valData.shape, valLabel.shape])
    print(x)

    model = model.fit(model.to(param.DEVICE), train_loader, val_loader, test_loader,param)
    return model
