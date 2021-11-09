from torch import nn
import torch
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import classification_report
from time import time
from datetime import datetime
import logging

dt_id = datetime.now().isoformat()
logging.basicConfig(filename=f'{dt_id}.log', level=logging.INFO)
logging.info(f'Start job {dt_id}')

model = resnet50(pretrained=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = ImageFolder(
    '/mnt/qb/datasets/ImageNet-C/speckle_noise/5',

    # DONT USE TRANSFORM BEFORE SPLIT?
    transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)


train_count = int(0.7 * len(dataset))
val_count = int(0.3 * len(dataset))
train_dataset, val_dataset = random_split(
    dataset, (train_count, val_count)
)
    
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=500)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=500)

model.eval()
def convert_to_latent(loader, max_length=80000):
    xo, xl, yl = [], [], []
    for samples, labels in loader:
        with torch.no_grad():
            res = model(samples)
            xo.extend(samples)
            xl.extend(res.numpy())

            yt = labels.numpy()
        yt[yt < 500] = -1
        yt[yt >= 500] = +1
        yl.extend(yt)
        if len(yl) >= max_length:
            break
    logging.info(f"Converted {len(xo)} datapoints to latent space.")
    return xo, np.array(xl), np.array(yl)

s = time()
train_latent = convert_to_latent(train_loader, 20_000)
val_latent = convert_to_latent(val_loader, 10_000)
logging.info(f'Elapsed: {time() - s} s')
# logging.info(f'Train latent: {sum([train_latent[i].size * train_latent[i].itemsize for i in range(len(train_latent))])/1024**2} MB')
# logging.info(f'Val latent:   {sum([val_latent[i].size * val_latent[i].itemsize for i in range(len(val_latent))])/1024**2} MB')


ksvm = SVC(kernel='rbf')
ksvm.fit(train_latent[1], train_latent[2])

true_y, pred_y = val_latent[2], ksvm.predict(val_latent[1])
logging.info(classification_report(true_y, pred_y, digits=3))


# def convert_to_imshow(image):
#     return np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)

def convert_to_imshow(tensor):
    std = torch.tensor(normalize.std).view(3, 1, 1)
    mean = torch.tensor(normalize.std).view(3, 1, 1)
    return TF.to_pil_image(torch.clamp(tensor * std + mean, 0, 1))

orig_i = np.argmax(val_latent[2])
logging.info(ksvm.predict(val_latent[1][orig_i:orig_i+1, :]))
plt.imshow(convert_to_imshow(val_latent[0][orig_i]))
plt.savefig(f'{dt_id}_selected.jpg')


km = rbf_kernel(ksvm.support_vectors_, val_latent[1][orig_i:orig_i+1, :])
km *= train_latent[2][ksvm.support_].reshape(len(ksvm.support_), 1)

def ksvm_label_flip(inds):
    prev = ksvm.predict(val_latent[1][orig_i:orig_i+1, :])[0]
    # Create coefficient backup
    backup = np.copy(ksvm._dual_coef_)
    # Flip labels
    ksvm._dual_coef_[0, inds] = -ksvm._dual_coef_[0, inds]
    # Check prediction
    after = ksvm.predict(val_latent[1][orig_i:orig_i+1, :])[0]
    # Restore coefficients
    ksvm._dual_coef_ = backup
    return prev, after

# Find the tipping point
label_flip_len = -1
inds = np.flip(km.argsort(axis=0).flatten())
beyond_end = 5
for i in range(1, len(inds)+1):
    p, a = ksvm_label_flip(inds[:i])
    if p != a:
        logging.info(f'Label flips: {i}')
        label_flip_len = i
        inds = inds[:label_flip_len + beyond_end]
        break

num_h = len(inds) // 4 + 2
plt.figure(figsize=(18, 35))
plt.subplot(num_h, 4, 1); plt.title('orig.'); plt.axis('off')
plt.imshow(convert_to_imshow(val_latent[0][orig_i]))
for i, ind in enumerate(inds):
    ti = ksvm.support_[ind]
    plt.subplot(num_h,4,i+2)
    plt.axis('off')
    plt.title(
        "{:.3f}{}".format(km[ind][0], 'y' if train_latent[2][ti] == 1 else 'n'), 
        fontdict={ 'color': 'red' if i >= label_flip_len else 'black' })
    plt.imshow(convert_to_imshow(train_latent[0][ti]))
plt.tight_layout()
plt.savefig(f'{dt_id}_sampled.jpg')
