import os
from datetime import datetime
from model.cascadeMEM import CMmodel
from tool.config_gen import base_config, gen_base_config
from tool.evaluate import evaluate
from tool.plugin import dict2obj
from tool.train import train

config = gen_base_config(base_config, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))

param = dict2obj(config)
if not os.path.exists(param.OUTPUT_PATH):
    os.makedirs(param.OUTPUT_PATH)

model = CMmodel(param.BACKBONE_GEN_FUNC(param.BACKBONE_SETTINGS), param=param)

trained = train(model, dict2obj(config))

result = evaluate(trained, dict2obj(config))
