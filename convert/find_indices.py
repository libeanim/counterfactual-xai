import numpy as np
from time import time
import torch
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

assert torch.cuda.is_available()
print('CUDA is availbale', torch.cuda.is_available())

split = "train"
print('USE SPLIT', split)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = ImageFolder(
    f'/mnt/qb/datasets/ImageNet2012/{split}',
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)

loader = DataLoader(dataset, shuffle=False, batch_size=64, num_workers=1)
assert loader.batch_size is not None

model = resnet50(pretrained=True)
model.eval()
model.cuda()

s = time()
batch_counter = 0
wrong_indices = []
with torch.inference_mode():
    for samples, labels in loader:
        outputs = model(samples.cuda())
        pred_y = torch.argmax(outputs, dim=1).cpu()
        inds = np.arange(batch_counter * loader.batch_size, batch_counter * loader.batch_size + labels.size()[0])
        wrong_indices.extend(inds[pred_y != labels])
        batch_counter += 1
        if batch_counter % 40 == 0:
            print(f'Search in progress ({batch_counter})')


print(f'Elapsed: {time() - s} s')
percentage_wrong = (len(wrong_indices) / (batch_counter * loader.batch_size)) * 100
print(f'Wrong images {percentage_wrong}%')
wrong_indices = np.array(wrong_indices)
np.save(f'/mnt/qb/work/bethge/ahochlehnert48/results/resnet50_wrong_{split}_indices.npy', wrong_indices)
print('Saved.')
#!ls -lah /home/bethge/ahochlehnert48/results/