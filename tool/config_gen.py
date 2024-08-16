from tool.backbone_gen import FLinear
from tool.dataset_map import dataset_info

base_config = {
    'PIN_MEMORY': True,
    'SPILT_SHUFFLE': True,
    'LOADER_SHUFFLE': True,
    'DROP_LAST': False,
    'VAL_SIZE': 0.1,
    'SPILT_SEED': 2022,
    'NUM_WORKERS': 0,
    'EPOCHS': 2,
    'LR': 1e-3,
    'WD': 0,
    'RHO': 0,
    'BATCH_SIZE': 256,
    'TRAIN_BATCH_SIZE': 256,
    'VAL_BATCH_SIZE': 256,
    'PATIENCE': 10,
    'DEVICE': 'cuda:0',
    'DATASET': 'MNIST',
    'ANOMALY_ID': 2,
    'DISABLE_TQDM': True,
    'BACKBONE_GEN_FUNC': FLinear,
    'BACKBONE_SETTINGS': None,
    'EXP_GROUP': None,
    'CUSTOM_EXP_NAME': None,
    'MEM_SIZE': 50,
    'ERASER_PROB': 0.1,
    'EPSILON': 15,
    'MEM_SIZE2': 1000,
    'ERASER_PROB2': 0.5,
    'EPSILON2': 5,
}


def gen_base_config(origin_config: object, timestamp: object) -> object:
    origin_config['EXP_START_TIME'] = timestamp
    origin_config['DATA_SHAPE'] = dataset_info[origin_config['DATASET']]['shape']
    temp = ['EXP_START_TIME', 'DATASET', 'ANOMALY_ID']

    if origin_config['EXP_GROUP'] is None:
        origin_config['OUTPUT_PATH'] = "output/" + "_".join([str(origin_config[i]) for i in temp]) + "_{}".format(
            origin_config['BACKBONE_GEN_FUNC']().name)
    else:
        origin_config['OUTPUT_PATH'] = "output/" + "{}/".format(origin_config['EXP_GROUP']) + "_".join(
            [str(origin_config[i]) for i in temp]) + "_{}".format(
            origin_config['BACKBONE_GEN_FUNC']().name)

    if origin_config['CUSTOM_EXP_NAME'] is not None:
        origin_config['OUTPUT_PATH'] += origin_config['CUSTOM_EXP_NAME']

    new = origin_config
    return new
