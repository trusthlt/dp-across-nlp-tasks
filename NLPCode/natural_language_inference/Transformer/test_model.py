import torch
import torch.nn as nn
from torchtext.legacy import data
import torch.optim as optim
import math
from prep_data import split_and_cut, tokenizer, convert_to_int
from model import BERTNLIModel
import time
from train_eval import train, evaluate
from utils import EarlyStopping

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def preprocessing(EP, device):
    cls_token_idx = tokenizer.cls_token_id
    sep_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    
    #For sequence
    TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = split_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    pad_token = pad_token_idx,
                    unk_token = unk_token_idx)
    #For label
    LABEL = data.LabelField()
    #For Attention mask
    ATTENTION = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = split_and_cut,
                    preprocessing = convert_to_int,
                    pad_token = pad_token_idx)
    #For token type ids
    TTYPE = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = split_and_cut,
                    preprocessing = convert_to_int,
                    pad_token = 1)

    # Fields will help to map the column with the torchtext Field.
    fields = [('label', LABEL), ('sequence', TEXT), ('attention_mask', ATTENTION), ('token_type', TTYPE)]

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = f'{EP.dataset_dir}snli_1.0',
                                        train = f'{EP.dataset_dir}snli_1.0_train.csv',
                                        validation = f'{EP.dataset_dir}snli_1.0_dev.csv',
                                        test = f'{EP.dataset_dir}snli_1.0_test.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True)
    
    LABEL.build_vocab(train_data)

    #Create iterator
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = EP.BATCH_SIZE,
        sort_key = lambda x: len(x.sequence),
        sort_within_batch = False, 
        device = device)

    return LABEL, train_iterator, valid_iterator, test_iterator

def natural_language_inference(EP):
    torch.manual_seed(EP.SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('start preprocessing')
    LABEL, train_iterator, valid_iterator, test_iterator = preprocessing(EP, device)
    print('end preprocessing')

    #defining model
    OUTPUT_DIM = len(LABEL.vocab)
    model = BERTNLIModel(EP, OUTPUT_DIM).to(device)

    optimizer = optim.Adam(model.parameters(),lr=EP.LEARNING_RATE,eps=1e-6)
    criterion = nn.CrossEntropyLoss().to(device)

    # do the differential privacy stuff
    if EP.privacy:
        if EP.alpha == None:
            privacy_engine = PrivacyEngine(model,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        else:
            print(f'Setting the alpha to {EP.alpha}')
            privacy_engine = PrivacyEngine(model,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        alphas=[EP.alpha],
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        privacy_engine = privacy_engine.to(device)
        privacy_engine.attach(optimizer)
    
    model.load_state_dict(torch.load(f"{EP.output_dir}/{EP.output_file}{EP.SEED}.pt"))
    test_loss, test_acc, test_precision, test_recall, test_f1  = evaluate(model, test_iterator, criterion, device)

    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f}')
    f = open(f"{EP.output_dir}/{EP.output_file}{EP.SEED}.txt", "a")
    f.write(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | Test Prec.: {test_precision:.3f} | Test Rec.: {test_recall:.3f} | Test F1: {test_f1:.3f}\n')
    f.close()

def main(EP):
    natural_language_inference(EP)

#if __name__ == "__main__":
    