import torch
from torchvision import  transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop

# def croptop(image):
#     #print("Image size", image.size)
#     width, height = image.size
#     return crop(image, int(.09*height), 0, height, width)



def get_dataloader_preprocess(path: str, batch_size: int, image_size:int, num_workers:int):
    """"Image Dataloader that returns a path"""
    transform = transforms.Compose([
                    # transforms.Lambda(croptop),
                    transforms.RandomApply(torch.nn.ModuleList([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0,translate=(.1,.1)),
                    transforms.ColorJitter(brightness=(.9,1.1)),
                    transforms.RandomAffine(degrees=0,scale=(0.85, 1.15)),
                    ]), p=0.5),
                    #transforms.Normalize(mean=[ 0.406], std=[0.225]),
                    transforms.ToTensor(),
                    transforms.Resize(size=(image_size,image_size)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    dataset = ImageFolder(path, transform = transform)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = True, num_workers=num_workers, drop_last=False)

    return dataloader, len(dataset)


def get_covid_dataloader_valid(path: str, batch_size: int, image_size:int, num_workers:int):
    """"Image Dataloader that returns a path"""
    transform = transforms.Compose([
                    #transforms.Normalize(mean=[ 0.406], std=[0.225]),
                    transforms.ToTensor(),
                    transforms.Resize(size=(image_size,image_size)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    dataset = ImageFolder(path, transform = transform)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = True, num_workers=num_workers, drop_last=False)

    return dataloader, len(dataset)
