from torch import nn
import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms.functional as TF
from datetime import datetime

assert torch.cuda.is_available()
print('CUDA is availbale', torch.cuda.is_available())

# General output directory
RESULT_DIR = 'results/resnet50new'

# Generate id to reference files/plots/images
dt_id = datetime.now().isoformat()
print(f'ID: {dt_id}')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def convert_to_imshow(tensor):
    """
    Converts a tensor containing image data (RGB)
    into a suitable format for matplotlib.pyplot.imshow
    """
    if len(tensor.shape) == 1:
        tensor = tensor.reshape(3, 224, 224)
    tensor = torch.from_numpy(tensor)
    std = torch.tensor(normalize.std).view(3, 1, 1)
    mean = torch.tensor(normalize.std).view(3, 1, 1)
    return TF.to_pil_image(torch.clamp(tensor * std + mean, 0, 1))


def load_data(npz_file: str):
    """Load data latent data from npz-file"""
    arr = np.load(npz_file)
    return [arr['arr_0'], arr['arr_1'], arr['arr_2']]

def initialise_weights(k):
    """
    Initialise weights of the k-prototype vectors per class.
    
    Currently the prototype vectors are randomly chosen from
    the latent training samples of a specific class. This should
    be replaced by a more sophisticated clustering.

    If no latent space training sample of a class is present
    the weights are initialised uniformly random.
    """
    def build_init_m(m, k, arr, y):
        assert arr.shape[1] == k
        m[:, y*k:y*k + k] = arr
        return m

    df = np.random.uniform(size=[2048, k*1000])

    for y in np.unique(train_latent[3]):
        # Get all latentn samples of class y
        tmp = train_latent[1][train_latent[3] == y]
        # Randomly select k indices
        inds = np.random.randint(0, tmp.shape[0], k)
        # Build weights from selected samples
        df = build_init_m(df, k, tmp[inds].T, y)
    
    return df
    

class ResNet50NN(nn.Module):
    
    def __init__(self, k=2, pretrained=True, init_pl=None, **kwargs):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        
        
        self.pl = nn.Linear(2048, k * 1000)
        if init_pl is not None:
            assert init_pl.shape[1] == 2048
            self.pl.weight.data = init_pl

        self.ml = nn.MaxPool1d(k)

        
    def forward(self, x):
        
        x = self.backbone(x)
        x = self.pl(x)
        x = torch.squeeze(self.ml(x.unsqueeze(0)))
        
        return x

def get_accuracy(model, dataloader, cuda=True):
    accuracy, steps = 0, 0
    with torch.no_grad():
        for inputs, lables in dataloader:

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            predictions = torch.argmax(outputs, axis=1)

            accuracy += torch.sum(predictions == lables) / lables.shape[0]
            steps += 1
        accuracy = accuracy / steps
        if cuda:
            accuracy = accuracy.cpu()
    return accuracy.numpy()


# Load imagenet subset for weights initialisation
train_latent = load_data('/home/bethge/ahochlehnert48/results/imnet_train_latent.npz')
val_latent = load_data('/home/bethge/ahochlehnert48/results/imnet_val_latent.npz')
print(f'Train latent: { sum([train_latent[i].size * train_latent[i].itemsize for i in range(len(train_latent))])/1024**2 } MB')
print(f'Val latent:   { sum([val_latent[i].size * val_latent[i].itemsize for i in range(len(val_latent))])/1024**2 } MB')
print(f'Train latent: { train_latent[0].size * train_latent[0].itemsize/1024**2 } MB')


## simplify labels
train_bird_inds = (train_latent[2] >= 10) * (train_latent[2] <= 24)
train_new_labels = -np.ones_like(train_latent[2])
train_new_labels[train_bird_inds] = 1
train_latent.insert(2, train_new_labels)

print('Difference:', np.sum(train_latent[2]), 'of', np.size(train_latent[2]))

## simplify lables - not neccessary actually, left for compatibility reason
val_bird_inds = (val_latent[2] >= 10) * (val_latent[2] <= 24)
val_new_labels = -np.ones_like(val_latent[2])
val_new_labels[val_bird_inds] = 1
val_latent.insert(2, val_new_labels)

print('Difference:', np.sum(val_latent[2]), 'of', np.size(val_latent[2]))


import wandb
wandb.init(project='resnet-50-mpl', entity='libeanim')
# 2. Save model inputs and hyperparameters
config = wandb.config
config.id = dt_id
config.learning_rate = 0.001
config.momentum = 0.9
config.epochs = 10
config.log_steps = 100
config.k = 3

# Initialise weights
w = torch.Tensor(initialise_weights(config.k)).T
# w=None
model = ResNet50NN(k=config.k, init_pl=w)

# Delete latent data from memory and trigger garbage collector
del train_latent
del val_latent
import gc; gc.collect()


train_dataset = ImageFolder(
    '/mnt/qb/datasets/ImageNet2012/train',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)

val_dataset = ImageFolder(
    '/mnt/qb/datasets/ImageNet2012/val',
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)
    
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=64)


wandb.watch(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
model.cuda()

for epoch in range(config.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_accuracy = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        preditions = torch.argmax(outputs, axis=1)
        running_accuracy += (torch.sum(preditions == labels) / outputs.shape[0]).cpu().detach().numpy()
        if i % config.log_steps == config.log_steps - 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / config.log_steps))
            # test_accuracy = get_accuracy(model, val_loader)
            wandb.log({"loss": running_loss / config.log_steps, "accuracy": running_accuracy / config.log_steps })
            running_loss = 0.0
            running_accuracy = 0

    
    torch.save(model.state_dict(), f"{RESULT_DIR}/{config.id}_{epoch}.pth")
# model.cpu()
torch.save(model.state_dict(), f"{RESULT_DIR}/{config.id}_final.pth")
print('Finished Training')