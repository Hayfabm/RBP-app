#!/home/user/miniconda3/envs/torch-gpu/bin/python
import os
import time

import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + ".npy", allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    # init neptune logger
    run = neptune.init(
        project="sophiedalentour/RBP-app",
        name="RBP classification",
        tags=["pytorch", "neural_network"],
        capture_hardware_metrics=False,
    )
    # Load preprocessed data
    dir_input = "features/"
    features = load_tensor(dir_input + "features", torch.FloatTensor)
    labels = load_tensor(dir_input + "labels", torch.LongTensor)

    # Create a dataset and split it into train/dev/test
    dataset = list(zip(features, labels))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_test = split_dataset(dataset, 0.85)
    print(len(dataset_train))
    print(len(dataset_test))

    # Create fully connected neural network
    class NN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_size, 100)
            self.fc2 = nn.Linear(100, num_classes)
            self.log_softmax = F.log_softmax

        def forward(self, x):
            x = x.view(-1, x.size(0))
            x = F.relu(self.fc1(x))
            x = self.log_softmax(self.fc2(x), dim=1)
            return x

    # Hyperparameters
    input_size = 616
    num_classes = 2
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 20

    # initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train network

    for epoch in range(num_epochs):
        print("epochs:", epoch)
        train_loss = 0.0
        correct_train = 0
        total = len(dataset_train)
        for i, (x_train, y_train) in enumerate(dataset_train, 1):
            # Get data to cuda if possible
            x_train = x_train.to(device=device)
            y_train = y_train.to(device=device)
            x_train = x_train.reshape(x_train.shape[0], 1)
            # forward
            scores = model(x_train)
            loss = criterion(scores, y_train)
            _, predictions = scores.max(1)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam strep
            optimizer.step()

            # accuracy & loss
            train_loss += loss.item()
            correct_train += predictions.eq(y_train).sum().item()

            train_epoch_loss = train_loss / total
            train_epoch_acc = 100 * (correct_train / total)
        print(
            "training_loss: {:.2f} | training_acc: {:.2f}".format(
                train_epoch_loss, train_epoch_acc
            )
        )
        print("-" * 70)

    # Log batch loss
    run["training/batch/loss"].log(train_epoch_loss)
    run["training/batch/acc"].log(train_epoch_acc)

    # check accuracy on training & test to see how good our model

    correct_test = 0
    total = len(dataset_test)
    print("evaluting trained model ...")
    y_pred = []
    with torch.no_grad():
        for x_test, y_test in dataset_test:
            x_test = x_test.to(device=device)
            y_test = y_test.to(device=device)
            x_test = x_test.reshape(x_test.shape[0], 1)
            scores = model(x_test)
            loss = criterion(scores, y_test)
            _, predictions = scores.max(1)
            correct_test += (predictions == y_test).item()

        print(
            f"Got {correct_test} / {total} with accuracy {float(correct_test)/float(total)*100:.2f}"
        )

    run.stop()
