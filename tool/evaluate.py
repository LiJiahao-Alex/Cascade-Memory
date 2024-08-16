import torch
from torch.utils.data import DataLoader, TensorDataset

from tool.dataset_map import load_dataset, dataset_info
from tool.plugin import evalue, MinMax_norm, ZScore_norm


def evaluate(model, param):
    _, _, test_data, test_label = load_dataset(param.DATASET)
    test_label = (test_label == param.ANOMALY_ID)
    trans = [MinMax_norm, ZScore_norm]
    if dataset_info[param.DATASET]["type"] == "images":
        for f in trans:
            test_data = f(test_data)
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=param.BATCH_SIZE,
                             shuffle=False,
                             num_workers=param.NUM_WORKERS,
                             pin_memory=param.PIN_MEMORY,
                             drop_last=False
                             )
    result = evalue(model, test_loader, param)
    print("F1:{}".format(result['best_f1']))
    print("Thresh:{}".format(result['best_f1_thresh']))
    print("AP:{}".format(result['ap']))
    print("AUC:{}".format(result['auc']))
    return result
