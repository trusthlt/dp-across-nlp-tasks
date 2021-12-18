import os
import torch
import pandas as pd
import sys
import os
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Transformer.dataset.utils_squad import (read_squad_examples, convert_examples_to_features)
from pytorch_transformers import BertTokenizer
from torch.utils.data import TensorDataset

def helper_create_set(cached_features_file, data_set, tokenizer, is_test, EP, using_test_bench=False):
    if not using_test_bench:
        if not os.path.exists(cached_features_file):
            # Cache features for faster loading
            features = convert_examples_to_features(examples=data_set,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=EP.max_seq_length,
                                                    doc_stride=EP.doc_stride,
                                                    max_query_length=EP.max_query_length,
                                                    is_training=not is_test) # set it to true so the features have start and endtoken included
            torch.save(features, cached_features_file)
        else:
            features = torch.load(cached_features_file)
    else:
        # Cache features for faster loading
            features = convert_examples_to_features(examples=data_set,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=EP.max_seq_length,
                                                    doc_stride=EP.doc_stride,
                                                    max_query_length=EP.max_query_length,
                                                    is_training=not is_test) # set it to true so the features have start and endtoken included
    
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    if not is_test:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index, all_cls_index, all_p_mask, all_start_positions, all_end_positions)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index, all_cls_index, all_p_mask)
    
    return features, dataset

def load_train_set(tokenizer, EP):
    # train data
    input_file = f'{EP.dataset_dir}my_train.json'
    train_set = read_squad_examples(input_file=input_file,
                                    is_training=True,
                                    version_2_with_negative=True)

    train_data = pd.DataFrame.from_records([vars(example) for example in train_set])
    train_data['paragraph_len'] = train_data['doc_tokens'].apply(len)
    train_data['question_len'] = train_data['question_text'].apply(len)

    # val data
    input_file = f'{EP.dataset_dir}my_val.json'
    val_set = read_squad_examples(input_file=input_file,
                                    is_training=True,
                                    version_2_with_negative=True)

    val_data = pd.DataFrame.from_records([vars(example) for example in val_set])
    val_data['paragraph_len'] = val_data['doc_tokens'].apply(len)
    val_data['question_len'] = val_data['question_text'].apply(len)

    # create train and val split
    '''train_len = int(len(train_data) * 0.8)
    val_len = len(train_data) - train_len
    # split the dataframe
    val_data = train_data[:val_len]
    train_data = train_data[val_len:]

    # split the squad example
    val_set = train_set[:val_len]
    train_set = train_set[val_len:]

    # valid length is 26064
    # train length is 104255'''
    
    cached_features_file_train = f'{EP.dataset_dir}cached_features/cached_train'
    cached_features_file_val = f'{EP.dataset_dir}cached_features/cached_val'
    ##################
    # create Train set
    ##################
    train_features, train_dataset = helper_create_set(cached_features_file_train, train_set, tokenizer, False, EP)

    ################
    # create val set
    ################
    val_features, val_dataset = helper_create_set(cached_features_file_val, val_set, tokenizer, False, EP)


    # print(length)
    print(f'The train set length is {len(train_set)}')
    print(f'The val set length is {len(val_set)}')
    return train_dataset, train_set, train_features, val_dataset, val_set, val_features, EP.BATCH_SIZE/len(train_set)

def load_test_set(tokenizer, EP, input_file=None):
    # if we use the test bench
    if input_file == None:
        input_file = f'{EP.dataset_dir}test-v2.0.json'
    test_set = read_squad_examples(input_file=input_file,
                                    is_training=False,
                                    version_2_with_negative=True)
    
    cached_features_file = f'{EP.dataset_dir}cached_features/cached_test'
    using_test_bench = not input_file == None
    features, dataset = helper_create_set(cached_features_file, test_set, tokenizer, True, EP, using_test_bench)


    return dataset, test_set, features