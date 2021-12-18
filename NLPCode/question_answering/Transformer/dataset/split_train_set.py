import json
import re
# total number of data 442
train_size = 350
val_size = 92
path = "<path to json dataset file of SQuAD>"
with open(path) as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']

    train_json = {"data" : dataset[:train_size]}
    train_json = json.dumps(train_json)
    train_json = train_json.replace("False", "false")
    train_json = train_json.replace("True", "true")
    train_f = open("<path for new json dataset file>", 'w')
    train_f.write(str(train_json))
    train_f.close()

    val_json = {"data" : dataset[train_size:]}
    val_json = json.dumps(val_json)
    val_json = val_json.replace("False", "false")
    val_json = val_json.replace("True", "true")
    val_f = open("<path for new json dataset file>", 'w')
    val_f.write(str(val_json))
    val_f.close()