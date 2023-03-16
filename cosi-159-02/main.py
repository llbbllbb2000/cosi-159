import torch
import argparse
from train import Trainer
from torch.utils.data import DataLoader
from model import SphereFace
from dataset import LFWDataset
import eval


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch sphereface')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--eval', type=int, default=0, help="test")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    if args.eval != 0 :
        test_dataset = LFWDataset('pairsDevTrain.txt')
        test_loader = DataLoader(test_dataset, batch_size = args.bs, shuffle=True)
        eval.verify(test_loader)
        return

    # model
    model = SphereFace()

    # datasets
    train_dataset = LFWDataset('pairsDevTrain.txt')
    train_loader = DataLoader(train_dataset, batch_size = args.bs, shuffle=True)

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")
    # trainer.load_model("./save/model.pth")


if __name__ == "__main__":
    main()