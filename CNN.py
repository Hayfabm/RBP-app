#!/home/user/miniconda3/envs/torch-gpu/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime
import neptune.new as neptune


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

    # Create simple CNN
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(1, 616),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

            self.fc1 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = torch.sigmoid(self.fc1(x))

            return x

    # Hyperparameters
    in_channel = 1
    num_classes = 2
    learning_rate = 0.0001
    batch_size = 64
    num_epochs = 10

    params = {
        "lr": 0.001,
        "bs": 64,
        "input_sz": 616,
        "n_classes": 2,
        "model_filename": "model",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }
    model_path = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    # initialize network
    model = CNN().to(device)
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
            # x_train = x_train.reshape(x_train.shape[0], 1)
            x_train = torch.reshape(x_train, (1, 616)).to(device)
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

    # Log batch loss & acc
    run["training/batch/loss"].log(train_epoch_loss)
    run["training/batch/acc"].log(train_epoch_acc)
    run["config/hyperparameters"] = params

    # check accuracy on training & test to see how good our model

    correct_test = 0
    total = len(dataset_test)
    print("evaluting trained model ...")
    y_pred = []
    with torch.no_grad():
        for x_test, y_test in dataset_test:
            x_test = x_test.to(device=device)
            y_test = y_test.to(device=device)
            # x_test = x_test.reshape(x_test.shape[0], 1)
            x_test = torch.reshape(x_test, (1, 616)).to(device)
            scores = model(x_test)
            loss = criterion(scores, y_test)
            _, predictions = scores.max(1)
            correct_test += (predictions == y_test).item()
            test_epoch_acc = 100 * (correct_test / total)
            """
            precision = precision_score(predictions, y_test)
            print("Precision: %f" % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(predictions, y_test)
            print("Recall: %f" % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1 = f1_score(predictions, y_test)
            print("F1 score: %f" % f1)
            """

        print("accuracy: {:.2f}".format(test_epoch_acc))

        print(
            f"Got {correct_test} / {total} with accuracy {float(correct_test)/float(total)*100:.2f}"
        )
    run["testing/batch/acc"].log(test_epoch_acc)
    torch.save(model, model_path)
    run.stop()
