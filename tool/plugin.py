import copy
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def ZScore_norm(raw_data, mu=None, std=None, inverse=False):
    data = copy.deepcopy(raw_data)
    C = data.shape[1]
    if std is None:
        sigma = [0.5, 0.5, 0.5] if C == 3 else [0.5]
    if mu is None:
        mu = [0.5, 0.5, 0.5] if C == 3 else [0.5]
    for i, (mean, std) in enumerate(zip(mu, sigma)):
        if inverse:
            data[:, i, :, :] = data[:, i, :, :] * std + mean
        else:
            data[:, i, :, :] = (data[:, i, :, :] - mean) / std
    return data


def MinMax_norm(data, inverse=False):
    if inverse:
        return (data * 255).astype(int)
    else:
        return data / 255


def output_auc(model, test_loader, param):
    model.eval()
    temp = []
    temp2 = []
    temp3 = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.float().to(param.DEVICE)
            output = model(data)
            temp.append(output['recon'].cpu().numpy())
            temp2.append(data.cpu().numpy())
            temp3.append(label.cpu().numpy())
    recon = np.concatenate(temp)
    orign = np.concatenate(temp2)
    label = np.concatenate(temp3)
    res = recon - orign
    anomaly_score = np.linalg.norm(res.reshape(res.shape[0], -1), 2, axis=1)
    fpr, tpr, auc_thresh = roc_curve(label, anomaly_score)
    auroc = auc(fpr, tpr)
    return auroc


def evalue(model, test_loader, param):
    evaluate_result = {}
    model.eval()
    temp = []
    temp2 = []
    temp3 = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.float().to(param.DEVICE)
            output = model(data)
            temp.append(output['recon'].cpu().numpy())
            temp2.append(data.cpu().numpy())
            temp3.append(label.cpu().numpy())

    recon = np.concatenate(temp)
    orign = np.concatenate(temp2)
    label = np.concatenate(temp3)

    res = recon - orign

    anomaly_score = np.linalg.norm(res.reshape(res.shape[0], -1), 2, axis=1)

    fpr, tpr, auc_thresh = roc_curve(label, anomaly_score)
    auroc = auc(fpr, tpr)

    precision, recall, ap_thresh = precision_recall_curve(label, anomaly_score)
    ap = average_precision_score(label, anomaly_score)

    f1 = 2 * (precision * recall) / (precision + recall)

    best_f1_score = np.max(f1[np.isfinite(f1)])
    best_f1_score_index = np.argmax(f1[np.isfinite(f1)])

    best_thresh = ap_thresh[best_f1_score_index]

    evaluate_result['fpr'] = fpr
    evaluate_result['tpr'] = tpr
    evaluate_result['auc_thresh'] = auc_thresh
    evaluate_result['auc'] = auroc
    evaluate_result['precision'] = precision
    evaluate_result['recall'] = recall
    evaluate_result['ap_thresh'] = ap_thresh
    evaluate_result['ap'] = ap
    evaluate_result['f1'] = f1
    evaluate_result['best_f1'] = best_f1_score
    evaluate_result['best_f1_thresh'] = best_thresh
    return evaluate_result
