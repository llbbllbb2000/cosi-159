from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LFWDataset(Dataset) :
    def __init__(self, training_file):
        with open(training_file) as f :
            self.lines = f.readlines()[1:]

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) :
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip().split('\t')
        if len(line) == 3:
            # same identity
            same = True
            path1 = "./data/lfw-py/lfw_funneled/" + line[0]+'/'+line[0]+'_'+line[1].zfill(4)+'.jpg'
            path2 = "./data/lfw-py/lfw_funneled/" + line[0]+'/'+line[0]+'_'+line[2].zfill(4)+'.jpg'
        else:
            # different identities
            same = False
            path1 = "./data/lfw-py/lfw_funneled/" + line[0]+'/'+line[0]+'_'+line[1].zfill(4)+'.jpg'
            path2 = "./data/lfw-py/lfw_funneled/" + line[2]+'/'+line[2]+'_'+line[3].zfill(4)+'.jpg'

        img1 = Image.open(path1)
        img2 = Image.open(path2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, same
