import time
import torch
from torch import nn

from data_loader_torch import load_data


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.d1 = nn.Linear(in_features=31, out_features=100)
        self.relu1 = nn.ReLU()
        self.d2 = nn.Linear(in_features=100, out_features=100)
        self.relu2 = nn.ReLU()
        self.d3 = nn.Linear(in_features=100, out_features=100)
        self.relu3 = nn.ReLU()
        self.d4 = nn.Linear(in_features=100, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x1 = self.d1(input)
        x2 = self.relu1(x1)
        x3 = self.d2(x2)
        x4 = self.relu2(x3)
        x5 = self.d3(x4)
        x6 = self.relu3(x5)
        x7 = self.d4(x6)
        out = self.sig(x7)

        return out


def train(model, optimizer, criterion, n_epochs, train_loader, val_loader, patience=1, save_path='classifier.model',
          bucket_size=22):
    min_val_acc = float('inf') * -1
    epochs_no_improvement = 0

    total_time = 0.0
    for epoch in range(n_epochs):
        start_time = time.time()

        loss_train_epoch, loss_val_epoch = 0, 0

        model.train()

        acc_train = []
        acc_val = []
        length_train = 0
        length_val = 0

        for i, data in enumerate(train_loader):
            x, y = data

            # converting the data into GPU format
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            output_train = model(x)

            # clearing the Gradients of the model parameters
            optimizer.zero_grad()

            # computing the training and validation loss
            loss_train = criterion(output_train, y.unsqueeze(2))

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()

            loss_train_epoch += loss_train.item()

            acc_train.append(torch.sum(torch.round(output_train) == y.unsqueeze(2)))
            length_train += x.shape[0]

        loss_train_epoch = loss_train_epoch / len(train_loader)

        model.eval()

        with torch.no_grad():

            for i, data in enumerate(val_loader):
                x, y = data

                y[torch.where(y > 0)] = 1

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                output_val = model(x)
                loss_val = criterion(output_val, y.unsqueeze(2))
                loss_val_epoch += loss_val.item()

                acc_val.append(torch.sum(torch.round(output_val) == y.unsqueeze(2)))
                length_val += x.shape[0]

        loss_val_epoch = loss_val_epoch / len(val_loader)

        acc_train = (sum(acc_train) / (length_train * bucket_size)).item()
        acc_val = (sum(acc_val) / (length_val * bucket_size)).item()

        if acc_val >= min_val_acc:
            min_val_acc = loss_val_epoch
            epochs_no_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improvement += 1

        end_time = time.time()
        epoch_time = end_time - start_time
        total_time = total_time + epoch_time
        avg_time = total_time / (epoch + 1)

        # printing the training and validation loss
        print('Epoch : ', epoch + 1, '\t', 'train_loss :', loss_train_epoch, '\t', 'val_loss :', loss_val_epoch, '\t',
              "epoch time :", epoch_time, "s", 'accuracy train:', acc_train, '\t', 'accuracy validation:', acc_val)

        if epochs_no_improvement >= patience:
            print('Early stopping!')
            break
