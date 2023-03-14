import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import model


class Trainer:
    """ Trainer for SphereFace """

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
        criterion = model.AngleLoss()
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            cnt = 0
            for data, labels in train_loader:
                # print(data.shape)
                # print(target)
                # print()
                optimizer.zero_grad()
                outputs = self._model(data)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print(loss)
                # print(loss.data)
                train_loss += loss.item()
                _, predicted = torch.max(outputs[0], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                cnt += 1
                if cnt % 10 == 0 :
                    print("Loss: %.4f; Acc: %.5f" % (train_loss/(batch_idx + 1), correct/total))

            train_loss /= len(train_loader)

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.4f; Acc: %.5f" 
                  % (i + 1, epochs, elapse, train_loss, correct/total))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "sphereface_model.pth"))

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