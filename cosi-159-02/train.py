import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from model import AngleLoss


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
        criterion = AngleLoss()
        train_loss = 0
        self._model.train()

        print("Start training...")
        for T in range(epochs):
            tik = time.time()
            for i, (img1, img2, same) in enumerate(train_loader):
                label = torch.tensor([0 if s else 1 for s in same])
                optimizer.zero_grad()
                output1 = self._model(img1, label)
                output2 = self._model(img2, label)

                loss = criterion(output1, label) + criterion(output2, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.4f" 
                  % (T + 1, epochs, elapse, train_loss))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "sphereface_model.pth"))

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        self._model.load_state_dict(torch.load(path))