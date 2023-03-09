import argparse

import torch
import torchvision

from net_sphere import Net
import net_sphere

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch sphereface')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = ()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.LFWPeople(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.LFWPeople(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")
    # trainer.load_model("./save/mnist.pth")

    # model evaluation
    print("\nFor the test set, the accuracy is {:.5f}".format(trainer.eval(test_loader=test_loader)))

    # model inference
    x, y = next(enumerate(train_loader))[1]
    print(x.size(0))
    for i in range(x.size(0)):
        sample = x[i]  # complete the sample here
        print("predict:{}, true:{}".format(trainer.infer(sample=sample), y[i].item()))

    return


if __name__ == "__main__":
    main()