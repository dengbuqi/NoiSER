import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class GaussianImageDataset(Dataset):
    def __init__(self, length=2000, img_size=(104,104), mean=0, stddev=1, transform=None):
        self.transform = transform
        self.length = length
        self.img_size = img_size
        self.mean, self.stddev = mean, stddev
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.normal(self.mean, self.stddev, (3, *self.img_size))
        image = torch.clamp(image,-1,1)
        if self.transform is not None:
            image = self.transform(image)
        return image
    
if __name__=='__main__':
    print('Dataset demo!')
    dataset = GaussianImageDataset()
    to_pil_image((dataset[0]+1)/2).save(f'./train_input_demo.png')