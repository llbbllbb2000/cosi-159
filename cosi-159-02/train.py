import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import net_sphere


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        criterion = net_sphere.AngleLoss()
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            for data, target in train_loader:
                optimizer.zero_grad()
                outputs = self._model(data)

                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]
                outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).sum()


            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.4f Acc: %.5f" 
                  % (i + 1, epochs, elapse, train_loss/(batch_idx + 1), correct/total))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "model.pth"))

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """

        self._model.eval()

        correct = 0
        total = 0

        with torch.no_grad() :
            for data, target in test_loader :
                output = self._model(data)
                _, predict = torch.max(output.data, 1)
                total += data.size(0)
                correct += (predict == target).sum().item()

        return correct / total

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """

        self._model.eval()
        with torch.no_grad() :
            output = self._model(sample)
            _, predict = torch.max(output.data, 1)
            return predict.item()

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        self._model.load_state_dict(torch.load(path))

