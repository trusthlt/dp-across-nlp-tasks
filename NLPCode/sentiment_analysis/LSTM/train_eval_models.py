import torch
from utils import binary_accuracy
from tuning_structs import Privacy
from sklearn.metrics import precision_recall_fscore_support

only_one_iteration = False

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    precission = 0
    recall = 0
    f1 = 0
    model.train()

    for i, batch in enumerate(iterator):

        optimizer.zero_grad()
        text, text_lengths = batch.text

        predictions = model(text, text_lengths, device).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        pr_epoch, rec_epoch, f1_epoch, _ = precision_recall_fscore_support(batch.label.cpu().detach().numpy(), torch.round(torch.sigmoid(predictions)).cpu().detach().numpy(),average='macro')
        
        precission += pr_epoch
        rec_epoch += rec_epoch
        f1_epoch += f1_epoch

        if only_one_iteration:
            break        

    return epoch_loss / len(iterator), epoch_acc / len(iterator), precission / len(iterator), recall / len(iterator), f1 / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    precission = 0
    recall = 0
    f1 = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths, device).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            pr_epoch, rec_epoch, f1_epoch, _ = precision_recall_fscore_support(batch.label.cpu().detach().numpy(), torch.round(torch.sigmoid(predictions)).cpu().detach().numpy(), average='micro')
            precission += pr_epoch
            recall += rec_epoch
            f1 += f1_epoch

            if only_one_iteration:
                break

    return epoch_loss / len(iterator), epoch_acc / len(iterator), precission / len(iterator), recall / len(iterator), f1 / len(iterator)

