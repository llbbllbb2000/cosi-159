import argparse
import torchvision
import torch
from train import Trainer
from model import sphere4a


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch sphereface')
    parser.add_argument('--epochs', type=int, default=20, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # model
    model = sphere4a()

    # datasets
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize((112, 96)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.LFWPeople(root='./data/', download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")
    # trainer.load_model("./save/model.pth")


if __name__ == "__main__":
    main()