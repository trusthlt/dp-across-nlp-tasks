import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import time
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformer_model import BERTGRUSentiment
from train_eval_models import train, evaluate
from utils import epoch_time, EarlyStopping
from opacus import PrivacyEngine
from tuning_structs import Privacy, TrainingBERT, Tuning
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer


torch.backends.cudnn.deterministic = True
tokenizer = None
max_input_length = None



def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    # the transformer is trained on a fixed length so we need to cut the tokens to a maximum length
    tokens = tokens[:max_input_length - 2]
    return tokens


def data_preprocessing(EP, LABEL, train_data, test_data, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    
    print("splitting training set into train and validation")
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    print(f'so finally the length of the data: train={len(train_data)}, val={len(valid_data)}, test={len(test_data)}')

    # build vocab for the labels
    print("building vocab")
    LABEL.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("getting iterators")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=EP.BATCH_SIZE,
        device=device)

    return device, train_iterator, valid_iterator, test_iterator, EP.BATCH_SIZE/len(train_data)


def sentiment_analysis(EP, LABEL, train_data, test_data, SEED):
    print("---Starting Data Preprocessing---")
    device, train_iterator, valid_iterator, test_iterator, sample_rate = data_preprocessing(EP, LABEL, train_data, test_data, SEED)
    print("---Create Model---")
    if EP.use_BERT:
        tokenizer = BertTokenizer.from_pretrained(EP.bert_model_type)
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")
    
    model = BERTGRUSentiment(bert, EP.HIDDEN_DIM, EP.OUTPUT_DIM, EP.N_LAYERS, EP.BIDIRECTIONAL, EP.DROPOUT, EP.tuning, EP.use_rnn, EP.use_BERT)
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=EP.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    if EP.tuning.privacy:
        print("privacy enabled")

        if EP.alpha == None:
            privacy_engine = PrivacyEngine(model.trainable_layers,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)
        else:
            privacy_engine = PrivacyEngine(model.trainable_layers,
                                        sample_rate=sample_rate,
                                        target_delta= EP.delta,
                                        target_epsilon=EP.epsilon,
                                        noise_multiplier=EP.noise_multiplier,
                                        alphas=[EP.alpha],
                                        epochs =EP.N_EPOCHS,
                                        max_grad_norm=EP.max_grad_norm)

        privacy_engine.attach(optimizer)
        privacy_engine.to(device)

    criterion = criterion.to(device)

    best_valid_acc = -1
    early_stopping = EarlyStopping(EP.PATIENCE)

    print("---Start Training---")
    for epoch in range(EP.N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc, train_precission, train_recall, train_f1 = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc, valid_precission, valid_recall, valid_f1 = evaluate(model, valid_iterator, criterion, device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f'{EP.output_dir_model}/{EP.output_file}{SEED}.pt')
        
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s \n')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Prec.: {train_precission:.3f} | Train Rec.:{train_recall:.3f} | Train F1:{train_f1:.3f} \n')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}% | Valid Prec.: {valid_precission:.3f} | Valid Rec.:{valid_recall:.3f} | Valid F1:{valid_f1:.3f} \n')
        
        f = open(f'{EP.output_dir_txt}/{EP.output_file}{SEED}.txt', 'a')
        f.write(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s \n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Prec.: {train_precission:.3f} | Train Rec.:{train_recall:.3f} | Train F1:{train_f1:.3f} \n')
        f.write(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}% | Valid Prec.: {valid_precission:.3f} | Valid Rec.:{valid_recall:.3f} | Valid F1:{valid_f1:.3f} \n')
        f.close()

        # log privacy budget
        if EP.privacy:
            f = open(f'{EP.output_dir_txt}/{EP.priv_output}{SEED}.txt', 'a')
            f.write(f"Epoch: {epoch}\n")
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(EP.delta)
            nm = optimizer.privacy_engine.noise_multiplier
            print(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha}  and noise multiplier = {nm}")
            f.write(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha}  and noise multiplier = {nm}\n")
            f.close()

        # check for early stopping
        if EP.early_stopping_active and early_stopping.should_stop(valid_acc):
            print(f'Did early stoppin in epoch {epoch}')
            f = open(f'{EP.output_dir_txt}/{EP.output_file}{SEED}.txt', 'a')
            f.write(f'Did early stoppin in epoch {epoch}\n')
            f.close()
            break

    # test model
    model.load_state_dict(torch.load(f'{EP.output_dir_model}/{EP.output_file}{SEED}.pt'))
    test_loss, test_acc, test_precission, test_recall, test_f1 = evaluate(model, test_iterator, criterion, device)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Prec.: {test_precission:.3f} | Test Rec.:{test_recall:.3f} Test F1:{test_f1:.3f} \n')
    f = open(f'{EP.output_dir_txt}/{EP.output_file}{SEED}.txt', 'a')
    f.write(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Prec.: {test_precission:.3f} | Test Rec.:{test_recall:.3f} Test F1:{test_f1:.3f} \n')
    f.close()

    return test_loss, test_acc, test_precission, test_recall, test_f1

def get_dataset(EP):
    global tokenizer
    global max_input_length
    # define the tokenizer the bert model was trained on
    tokenizer = BertTokenizer.from_pretrained(EP.bert_model_type)
    max_input_length = tokenizer.max_model_input_sizes[EP.bert_model_type]
    # get all special tokens the transformer was trained on
    init_token_idx = tokenizer.cls_token_id # token that indicates the beginning of the sentence
    eos_token_idx = tokenizer.sep_token_id # token that indicates the end of the sentence
    pad_token_idx = tokenizer.pad_token_id # token used to fill shorter sentences to max size
    unk_token_idx = tokenizer.unk_token_id # unknown word token
    print("creating data object for preprocessing")
    TEXT = data.Field(batch_first=True,
                    use_vocab=False,
                    tokenize=tokenize_and_cut,
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=init_token_idx,
                    eos_token=eos_token_idx,
                    pad_token=pad_token_idx,
                    unk_token=unk_token_idx)
    print("getting Labels")
    LABEL = data.LabelField(dtype=torch.float)
    print("splitting datasets into train and test")
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=EP.dataset_dir)

    return LABEL, train_data, test_data


def main(EP):
    LABEL, train_data, test_data = get_dataset(EP)

    loss_arr = []
    acc_arr = []
    prec = []
    rec = []
    f1 = []

    for SEED in EP.seeds:
        loss, acc, test_precission, test_recall, test_f1 = sentiment_analysis(EP, LABEL, train_data, test_data, SEED)

        loss_arr.append(loss)
        acc_arr.append(acc)

        prec.append(test_precission)
        rec.append(test_recall)
        f1.append(test_f1)

    print(f'ave. loss: {np.mean(np.array(loss_arr)):.3f} | ave. acc: {np.mean(np.array(acc_arr)):.3f} | ave. prec.:{np.mean(np.array(prec)):.3f} | ave. rec.:{np.mean(np.array(rec)):.3f} | ave. f1:{np.mean(np.array(f1)):.3f} \n')



if __name__ == "__main__":
    pass
    