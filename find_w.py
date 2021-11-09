from torch import nn
import torch
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import f1_score
from datetime import datetime

from scipy.optimize import minimize

dt_id = datetime.now().isoformat()
print(f'ID: {dt_id}')

def convert_to_imshow(tensor):
    if len(tensor.shape) == 1:
        tensor = tensor.reshape(3, 224, 224)
    tensor = torch.from_numpy(tensor)
    std = torch.tensor(normalize.std).view(3, 1, 1)
    mean = torch.tensor(normalize.std).view(3, 1, 1)
    return TF.to_pil_image(torch.clamp(tensor * std + mean, 0, 1))


def load_data(npz_file):
    arr = np.load(npz_file)
    return [arr['arr_0'], arr['arr_1'], arr['arr_2']]

class Identity(nn.Module):
    def forward(self, inputs):
        return inputs

model = resnet50(pretrained=True)
# model.fc = Identity()
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


## LOAD DATA
train_latent = load_data('/home/bethge/ahochlehnert48/results/imnet_train_latent.npz')
val_latent = load_data('/home/bethge/ahochlehnert48/results/imnet_val_latent.npz')
print(f'Train latent: { sum([train_latent[i].size * train_latent[i].itemsize for i in range(len(train_latent))])/1024**2 } MB')
print(f'Val latent:   { sum([val_latent[i].size * val_latent[i].itemsize for i in range(len(val_latent))])/1024**2 } MB')
print(f'Train latent: { train_latent[0].size * train_latent[0].itemsize/1024**2 } MB')


## DINO
# DINO_PATH = f'/home/bethge/sschneider/robustness-internal/dino-internal-2/features/imagenet/deit_small/8'
# DINO_PATH = f'/mnt/dino/deit_small/8'
# t = torch.load(f'{DINO_PATH}/trainfeat.pth')
# l = torch.load(f'{DINO_PATH}/trainlabels.pth')
# assert np.all(train_latent[2] == l[:train_latent[2].shape[0]].numpy())
# train_latent[1] = t[:train_latent[2].shape[0]].numpy()

# t = torch.load(f'{DINO_PATH}/testfeat.pth')
# l = torch.load(f'{DINO_PATH}/testlabels.pth')
# assert np.all(val_latent[2] == l[:val_latent[2].shape[0]].numpy())
# val_latent[1] = t[:val_latent[2].shape[0]].numpy()


## PREPROCESSING

## simplify labels
train_bird_inds = (train_latent[2] >= 10) * (train_latent[2] <= 24)
train_new_labels = -np.ones_like(train_latent[2])
train_new_labels[train_bird_inds] = 1
train_latent.insert(2, train_new_labels)

print('Difference:', np.sum(train_latent[2]), 'of', np.size(train_latent[2]))

## simplify lables
val_bird_inds = (val_latent[2] >= 10) * (val_latent[2] <= 24)
val_new_labels = -np.ones_like(val_latent[2])
val_new_labels[val_bird_inds] = 1
val_latent.insert(2, val_new_labels)

print('Difference:', np.sum(val_latent[2]), 'of', np.size(val_latent[2]))


## KNN

RESULT_DIR = '/home/bethge/ahochlehnert48/results/knn'


w = np.ones((1, train_latent[1].shape[1]))


def newc(params, metric='euclidean'):
    clf = KNeighborsClassifier(n_neighbors=1, metric=metric)
    clf.fit(params * train_latent[1], train_latent[2])
    pred_y = clf.predict(val_latent[1])
    return f1_score(val_latent[2], pred_y), clf


def f(params):
    f1, _ = newc(params)
    return -f1

res = minimize(f, w, method="dogleg",
               options={"disp": True, "return_all": False})
np.save(f'{RESULT_DIR}/{dt_id}_w.npy', res.x)