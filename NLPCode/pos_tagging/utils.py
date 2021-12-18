import torch
from sklearn.metrics import precision_recall_fscore_support
def categorical_accuracy(preds, y, tag_pad_idx, device):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    # F1 etc
    y_pred = max_preds[non_pad_elements].squeeze(1).squeeze(1)
    y_true = y[non_pad_elements].squeeze(1)
    pr_epoch, rec_epoch, f1_epoch, _ = precision_recall_fscore_support(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='macro')

    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device), pr_epoch, rec_epoch, f1_epoch

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

dataset_choice = {'EWT':'en_ewt', 'ESL':'en_cesl', 'GUM':'en_gum'}