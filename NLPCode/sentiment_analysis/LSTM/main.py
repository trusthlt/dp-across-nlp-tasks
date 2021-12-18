import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import torch.optim as optim
import torch.nn as nn
import time
from train_eval_models import train, evaluate
from lstm_model import LSTM
from utils import epoch_time, EarlyStopping
from opacus import PrivacyEngine
from opacus.utils import module_modification
from tuning_structs import Privacy
import numpy as np


def process_data(EP, train_data, test_data, TEXT, LABEL, SEED):

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    print("splitting training set into train and validation")
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    print(f'so finally the length of the data: train={len(train_data)}, val={len(valid_data)}, test={len(test_data)}')

    print("building vocab")
    TEXT.build_vocab(train_data,
                     max_size=EP.MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",  # pretraining (for no pretraining comment out)
                     unk_init=torch.Tensor.normal_)  # init vectors not with 0 but randomly guassian distributed
    LABEL.build_vocab(train_data)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("getting iterators")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=EP.BATCH_SIZE,
        sort_within_batch=True,
        device=device)

    return device, train_iterator, valid_iterator, test_iterator, EP.BATCH_SIZE / len(train_data)


def sentiment_analysis(EP, train_data, test_data, TEXT, LABEL, SEED):
    print("---Starting Data Preprocessing---")
    device, train_iterator, valid_iterator, test_iterator, sample_rate = process_data(EP, train_data, test_data, TEXT, LABEL, SEED)
    print("---Create Model---")

    # create model
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    INPUT_DIM = len(TEXT.vocab)
    model = LSTM(INPUT_DIM, EP.EMBEDDING_DIM, EP.HIDDEN_DIM, EP.OUTPUT_DIM, EP.N_LAYERS, EP.BIDIRECTIONAL, EP.DROPOUT, PAD_IDX, EP.privacy)

    # replace the initial weights of the embedding layer with the pretrained one
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # setting the <unk> and <pad> to zero since they have been inited using N(0,1), doing so tells the model that they
    # are irelevant to the learning of sentiment
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # get the index of <unk>
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EP.EMBEDDING_DIM)  # sets the embedding weights responsible for the
    # influence of the two tags to 0
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EP.EMBEDDING_DIM)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    print("THE SAMPLE RATE IS: " + str(sample_rate))
    
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

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Prec.: {train_precission:.3f} | Train Rec.:{train_recall:.3f} | Train F1:{train_f1:.3f} \n')
        print(f'\tValid Loss: {train_loss:.3f} | Valid Acc: {train_acc * 100:.2f}% | Valid Prec.: {train_precission:.3f} | Valid Rec.:{train_recall:.3f} | Valid F1:{train_f1:.3f} \n')
        
        f = open(f'{EP.output_dir_txt}/{EP.output_file}{SEED}.txt', 'a')
        f.write(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Prec.: {train_precission:.3f} | Train Rec.:{train_recall:.3f} | Train F1:{train_f1:.3f} \n')
        f.write(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}% | Valid Prec.: {valid_precission:.3f} | Valid Rec.:{valid_recall:.3f} | Valid F1:{valid_f1:.3f} \n')
        f.close()

        # log privacy budget
        if EP.privacy:
            f = open(f'{EP.output_dir_txt}/{EP.priv_output}{SEED}.txt', 'a')
            f.write(f"Epoch: {epoch}\n")
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(EP.delta)
            nm = optimizer.privacy_engine.noise_multiplier
            print(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}")
            f.write(f"(ε = {epsilon:.2f}, δ = {EP.delta}) for α = {best_alpha} and noise multiplier = {nm}\n")
            f.close()

        # check for early stopping
        if EP.early_stopping_active and early_stopping.should_stop(valid_acc):
            print(f'Did early stoppin in epoch {epoch}')
            
            f = open(f'{EP.output_dir_txt}/{EP.output_file}{SEED}.txt', 'a')
            f.write(f'Did early stoppin in epoch {epoch}\n')
            f.close()
            break

    model.load_state_dict(torch.load(f'{EP.output_dir_model}/{EP.output_file}{SEED}.pt'))

    # test the model
    test_loss, test_acc, test_precission, test_recall, test_f1 = evaluate(model, test_iterator, criterion, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Prec.: {test_precission:.3f} | Test Rec.:{test_recall:.3f} | Test F1:{test_f1:.3f} \n')
    
    f = open(f'{EP.output_dir_txt}/{EP.output_file}{SEED}.txt', 'a')
    f.write(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Prec.: {test_precission:.3f} | Test Rec.:{test_recall:.3f} | Test F1:{test_f1:.3f} \n')
    f.close()

    return test_loss, test_acc, test_precission, test_recall, test_f1

# download the dataset takes time so only do it once since it is not effected by the SEED
def get_dataset(EP):
    # get the dataset once
    print("creating data object for preprocessing")
    TEXT = data.Field(tokenize='spacy',
                      tokenizer_language=EP.spacy_english_model,
                      include_lengths=True)  # pytorch pads all sequences so they have equal length, now only the
    # elements of the sequence that are not padding are part of the learning
    print("getting Labels")
    LABEL = data.LabelField(dtype=torch.float)
    print("splitting datasets into train and test")
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=EP.dataset_dir)

    return train_data, test_data, TEXT, LABEL


def main(EP):
    train_data, test_data, TEXT, LABEL = get_dataset(EP)

    loss_arr = []
    acc_arr = []
    prec = []
    rec = []
    f1 = []
    
    for SEED in EP.seeds:
        loss, acc, test_precission, test_recall, test_f1 = sentiment_analysis(EP, train_data, test_data, TEXT, LABEL, SEED)
        loss_arr.append(loss)
        acc_arr.append(acc)

        prec.append(test_precission)
        rec.append(test_recall)
        f1.append(test_f1)

    print(f'ave. loss: {np.mean(np.array(loss_arr)):.3f} | ave. acc: {np.mean(np.array(acc_arr)):.3f} | ave. prec.:{np.mean(np.array(prec)):.3f} | ave. rec.:{np.mean(np.array(rec)):.3f} | ave. f1:{np.mean(np.array(f1)):.3f} \n')

if __name__ == "__main__":
    pass