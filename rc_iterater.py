from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

model_name = 'bert'
dataset = 'THUCNews'
x = import_module('models.' + model_name)
config = x.Config(dataset)

train_data, dev_data, test_data = build_dataset(config)
# train_iter = build_iterator(train_data, config)
# dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
print(test_iter)