import os
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

class LFWDataset(Dataset) :
    def __init__(self, training_file) -> None:
        with open(training_file) as f :
            self.class_num = int(f.readline())
            self.images = []
            # Read the match pairs
            for i in range(self.class_num) :
                words = f.readline().strip().split()
                self.images += [
                    (os.path.join("./lfw", words[0], words[0] + "_%04d.jpg" % words[1]))
                ] + [
                    (os.path.join("./lfw", words[0], words[0] + "_%04d.jpg" % words[2]))
                ]
            
            # Read the mismatch pairs
            for i in range(self.class_num) :
                words = f.readline().strip().split()
                self.images += [
                    (os.path.join("./lfw", words[0], words[0] + "_%04d.jpg" % words[1]))
                ] + [
                    (os.path.join("./lfw", words[2], words[2] + "_%04d.jpg" % words[3]))
                ]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __len__(self) :
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = cv2.imread(image_path, 1)

        return self.transform(image), label



