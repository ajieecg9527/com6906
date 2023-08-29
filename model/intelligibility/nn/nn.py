""" Build a Feedforward Neural Network. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class FeedforwardNeuralNetwork(nn.Module):
    """Class to represent a Neural Network of 2 fully connected layers with the sigmoid activation function.
    Fits a mathematical relation from input values (x1, x2) to output values y."""

    # params: np.ndarray | None = None  # The model params

    def __init__(self, n_inputs=3, n_neurons=64):  # n_neurons is the number of the neurons in the hidden layer
        super(FeedforwardNeuralNetwork, self).__init__()
        self.l1 = nn.Linear(n_inputs, n_neurons)  # from the input layer to the hidden layer
        self.l2 = nn.Linear(n_neurons, 1)  # from the hidden layer to the output layer

        self.optimizer = None
        self.criterion = None
        self.batch_size = 1  # by default
        self.num_workers = 1  # by default

    def forward(self, x):  # forward pass
        x = F.tanh(self.l1(x))  # mapping from score -> signal noise ratio?
        x = 100 * F.sigmoid(self.l2(x))  # correctness (0, 100)

        return x

    def create_data_loader(self, dataset=None, shuffle=True):
        if dataset is None:
            raise TypeError("The parameter DATASET is NoneType!!!")

        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers
        )

    def fit(self, train_set=None, dev_set=None, epochs=5, lr=0.0001, momentum=0.9, batch_size=1, num_workers=1):
        if train_set is None:
            raise TypeError("The parameter TRAIN_SET is NoneType!!!")
        if dev_set is None:
            raise TypeError("The parameter DEV_SET is NoneType!!!")

        # Create a data loader for the train set, and shuffle in each epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        train_loader = self.create_data_loader(train_set, shuffle=True)
        dev_loader = self.create_data_loader(dev_set, shuffle=False)

        # Set optimizer and loss function
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.MSELoss(reduction="mean")

        # Train the model on the train set
        for epoch in range(epochs):
            train_loss, dev_loss = 0.0, 0.0

            # Batch training
            for records in tqdm(train_loader, 0):
                inputs, labels = records[:, 1:], records[:, 0]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass, backward pass, and optimize
                outputs = self(inputs)
                mse_loss = self.criterion(outputs, labels)
                # rmse_loss = torch.sqrt(mse_loss)
                mse_loss.backward()
                self.optimizer.step()

                train_loss += mse_loss * len(records)

            # Valid
            for records in tqdm(dev_loader, 0):
                inputs, labels = records[:, 1:], records[:, 0]

                # Forward pass
                outputs = self(inputs)
                mse_loss = self.criterion(outputs, labels)

                dev_loss += mse_loss * len(records)

            train_loss = torch.sqrt(train_loss / len(train_loader))
            dev_loss = torch.sqrt(dev_loss / len(dev_loader))
            print(f"Epoch {epoch + 1}, train loss: {train_loss}, dev loss: {dev_loss}")

    def predict(self, test_set=None):
        if test_set is None:
            raise TypeError("The parameter TEST_SET is NoneType!!!")

        # Create a data loader for the test set
        self.batch_size = len(test_set)
        test_loader = self.create_data_loader(test_set, shuffle=False)

        # Predict (only one iteration)
        outputs, labels = None, None
        for records in tqdm(test_loader, 0):
            inputs, labels = records[:, 1:], records[:, 0]
            outputs = self(inputs)
            break

        return outputs
