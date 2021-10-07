"""
A module to define and run a model for next user cart prediction

Methods:
    train_model: define a model and run it on the data to learn
        a hidden layer, that holds information about past cart items;
    predict: make a prediction on one history matrix
        to get a cart vector of binary values, which correspond to
        category_ids in a user cart vector.
"""

__all__ = ["train_model", "predict"]


import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


MASK = 0 # value to add for vector padding


class CartGenerationModel(nn.Module):
    """ Recurrent NN to learn cart items history"""

    def __init__(self, embed_size, hidden_size):
        """ RNN initialization

        Attributes:
            embed_size: int, history length and length of features
            hidden_size: int, size of an internal memory matrix of RNN

        """
        super(CartGenerationModel, self).__init__()
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True)
        self.decoder = nn.Linear(
            in_features=hidden_size,
            out_features=1)

    def forward(self, matrix):
        """ Put a tensor of size Batch * Seq * Features
        through the GRU unit and transform the feature dim
        (or history) to 1 to get a vector prediction."""
        output, _ = self.rnn(matrix)
        output = self.decoder(output).squeeze()
        output = torch.sigmoid(output)
        return output

def _generate_batch(batch):
    """ Transform numpy matrices to torch tensors
    and pad on second dimension (user cart length or seq)
    to the longest in the batch """
    data = [torch.tensor(entry[0], dtype=torch.float32)
            for entry in batch]
    data = pad_sequence(data, batch_first=True, padding_value=MASK)
    target = [torch.tensor(entry[1], dtype=torch.float32)
              for entry in batch]
    target = pad_sequence(target, batch_first=True, padding_value=MASK)
    return data, target

def _get_f1_score(target, pred):
    """ Calculate f1 score for target and prediction vectors"""
    true_pos = (target * pred).sum(axis=1)
    true_neg = ((1 - target) * (1 - pred)).sum(axis=1)
    false_pos = ((1 - target) * pred).sum(axis=1)
    false_neg = (target * (1 - pred)).sum(axis=1)
    eps = 1e-7
    pres = true_pos / ( true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    f1 = 2 * pres * recall / (pres + recall + eps)
    return f1

def _define_model(dataset, hidden_size, learning_rate):
    """ Create a NN class object and define training method """
    model = CartGenerationModel(
        embed_size=dataset.get_feature_dim(),
        hidden_size=hidden_size
    )
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=learning_rate
    )
    return model, criterion, optimizer

def train_model(dataset, args):
    """ Train the RNN model on full history data
    and inform on the process if verbose is set to True"""
    model, criterion, optimizer = _define_model(
        dataset,
        args.hidden_size,
        args.learning_rate
    )
    model.train()
    train_losses = list()
    train_f1_accuracy = list()
    for epoch in range(args.num_epochs):
        start_time = time.time()
        losses = list()
        f1_accuracy = list()
        total = 0
        for data, target in DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=_generate_batch
        ):
            # zero the gradients from the previous step
            model.zero_grad()
            optimizer.zero_grad()

            # run the batch through the model
            # and make a gradient step
            pred = model(data)
            criterion(pred, target).backward()
            optimizer.step()

            # output of the RNN is a result of applying
            # the sigmoid function. To transform it to a binary result
            # the items compared to a threshold value
            threshold = torch.tensor([0.5])
            pred = (pred > threshold).float()

            # calculate the loss and a score on this batch and step
            avg_loss = criterion(pred, target).item() / data.shape[0]
            losses.append(avg_loss)
            f1_score = _get_f1_score(target, pred)
            avg_f1_score = f1_score.sum() / data.shape[0]
            f1_accuracy.append(avg_f1_score)

            # count the iteration
            total += 1

        # calculate end metrics
        epoch_loss = sum(losses) / total
        epoch_f1_accuracy = sum(f1_accuracy) / total
        train_losses.append(epoch_loss)
        train_f1_accuracy.append(epoch_f1_accuracy)

        # inform if necessary
        if args.verbose:
            secs = int(time.time() - start_time)
            mins = secs // 60
            secs = secs % 60
            print(
                'Train f1 accuracy {:.3f} in {} minutes {} seconds'.format(
                    epoch_f1_accuracy,
                    mins,
                    secs)
            )
            print(
                'Train loss {:.3f} in {} minutes {} seconds'.format(
                    epoch_loss,
                    mins,
                    secs)
            )
    return model

def predict(dataset, model, args):
    """ Make a cart prediction"""
    model.eval()
    dataset.change_mode()
    full_prediction = pd.DataFrame()
    with torch.no_grad():
        start_time = time.time()
        for user in dataset.get_users():
            # get the history and full cart category ids
            data, category_ids = dataset[user]

            # predict
            data = torch.tensor(data).float().unsqueeze(0)
            target = model(data)

            # transform to binary result
            threshold = torch.tensor([0.5])
            target = (target.cpu() > threshold).int().numpy()

            # create a DataFrame for this user
            prediction = pd.DataFrame({'id': category_ids, 'target': target})
            if len(full_prediction) == 0:
                full_prediction = prediction
            else:
                full_prediction = pd.concat([full_prediction, prediction])
        secs = int(time.time() - start_time)
        mins = secs // 60
        secs = secs % 60
        if args.verbose:
            print('Prediction done in {} mins {} sec'.format(mins, secs))
    return full_prediction
