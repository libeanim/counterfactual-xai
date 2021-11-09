import os
import sys
import json
import torch
from torch import nn
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


# Generate id to reference files/plots/images
dt_id = datetime.now().isoformat()
print(f'ID: {dt_id}')

# General output directory
RESULT_DIR = f'/home/bethge/ahochlehnert48/results/resnet50new/{dt_id}'
os.mkdir(RESULT_DIR)
print('Result dir:', RESULT_DIR)


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

# def initialise_weights(k):
#     """
#     Initialise weights of the k-prototype vectors per class.
    
#     Currently the prototype vectors are randomly chosen from
#     the latent training samples of a specific class. This should
#     be replaced by a more sophisticated clustering.

#     If no latent space training sample of a class is present
#     the weights are initialised uniformly random.
#     """
#     def build_init_m(m, k, arr, y):
#         assert arr.shape[1] == k
#         m[:, y*k:y*k + k] = arr
#         return m

#     df = np.random.uniform(size=[2048, k*1000])

#     for y in np.unique(train_latent[3]):
#         # Get all latentn samples of class y
#         tmp = train_latent[1][train_latent[3] == y]
#         # Randomly select k indices
#         inds = np.random.randint(0, tmp.shape[0], k)
#         # Build weights from selected samples
#         df = build_init_m(df, k, tmp[inds].T, y)
    
#     return df
    

class ResNet50NN(nn.Module):
    
    def __init__(self,
            k:int = 2, pretrained:bool = True,
            init_readout_weights:torch.Tensor = None,
            init_readout_bias: torch.Tensor = None,
            freeze_backbone: bool = False, **kwargs):
        """
        :param k: (int) number of readout vectors
        :param pretrained: (bool) use pretrained backbone
        :param init_readout_weights: (numpy.array) 
        :param init_readout_bias: (numpy.array)
        :param freeze_backbone: (bool) free backbone during training
        """
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        
        self.readout = nn.Linear(2048, k * 1000)
        if init_readout_weights is not None:
            assert init_readout_weights.shape[1] == 2048
            self.readout.weight.data = init_readout_weights
        if init_readout_bias is not None:
            # assert init_readout_bias.shape[1] == 2048
            self.readout.bias.data = init_readout_bias

        self.max_pool = nn.MaxPool1d(k)

        
    def forward(self, x):
        
        x = self.backbone(x)
        x = self.readout(x)
        x = torch.squeeze(self.max_pool(x.unsqueeze(0)))
        
        return x

def get_accuracy(model, loader, cuda=True):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.inference_mode():
        for data in loader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# LOAD CONFIG
config = dict(
    id = dt_id,
    random_seed = 42,
    model = "resnet50",
    learning_rate = 0.001,
    # learning_rate = 0.5,
    momentum = 0.9,
    epochs = 10,  # 18
    log_steps = 100,
    # k = 5,
    weight_decay = 1e-4,
    batch_size = 64,
    base_model = None,
    # base_model = "/home/bethge/ahochlehnert48/results/resnet50new/2021-09-21T10:29:43.000481_final.pth",

    # Scheduler step size
    # step_size = 100,
    # scheduler = 'MultiStepLR',
    # milestones = [30, 60, 90, 130, 150], #[10, 20, 30, 40, 50, 60]#[15, 30, 60, 90, 110, 140]
    # gamma = 0.1,
    # freeze_backbone = False,
    init_readout = None,
)
if len(sys.argv) == 2:
    cfg_path = sys.argv[1]
    if cfg_path == '--input':
        new_config = json.loads(input('Paste JSON config: '))
    elif os.path.exists(cfg_path):
        print('Reading config file:', cfg_path)
        with open(cfg_path, 'r') as cfg_file:
            new_config = json.load(cfg_file)
    else:
        new_config = json.loads(cfg_path)
    config.update(new_config)
print('Config:', json.dumps(config, indent=2))
# # Load imagenet subset for weights initialisation
# train_latent = load_data('/home/bethge/ahochlehnert48/results/imnet_train_latent.npz')
# val_latent = load_data('/home/bethge/ahochlehnert48/results/imnet_val_latent.npz')
# print(f'Train latent: { sum([train_latent[i].size * train_latent[i].itemsize for i in range(len(train_latent))])/1024**2 } MB')
# print(f'Val latent:   { sum([val_latent[i].size * val_latent[i].itemsize for i in range(len(val_latent))])/1024**2 } MB')
# print(f'Train latent: { train_latent[0].size * train_latent[0].itemsize/1024**2 } MB')


# ## simplify labels
# train_bird_inds = (train_latent[2] >= 10) * (train_latent[2] <= 24)
# train_new_labels = -np.ones_like(train_latent[2])
# train_new_labels[train_bird_inds] = 1
# train_latent.insert(2, train_new_labels)

# print('Difference:', np.sum(train_latent[2]), 'of', np.size(train_latent[2]))

# ## simplify lables - not neccessary actually, left for compatibility reason
# val_bird_inds = (val_latent[2] >= 10) * (val_latent[2] <= 24)
# val_new_labels = -np.ones_like(val_latent[2])
# val_new_labels[val_bird_inds] = 1
# val_latent.insert(2, val_new_labels)

# print('Difference:', np.sum(val_latent[2]), 'of', np.size(val_latent[2]))

torch.manual_seed(config["random_seed"])

# Save config

with open(f'{RESULT_DIR}/config.txt', 'w') as config_file: 
    # config_file.write(str(config.__dict__))
    json.dump(config, config_file)


# Initialise weights
if config["init_readout"] == 'fc_noisy':
    print('---> noisy fc')
    last_layer = resnet50(pretrained=True).fc
    w = np.repeat(last_layer.weight.data.numpy(), repeats=config["k"], axis=0)
    b = np.repeat(last_layer.bias.data.numpy(), repeats=config["k"], axis=0)
    w += 0.1 * w * np.random.uniform(size=[config["k"]*1000, 2048])
    b += 0.1 * b * np.random.uniform(size=[config["k"]*1000])
# w = torch.Tensor(initialise_weights(config["k"])).T
else:
    w = None
    b = None
if config['model'] == "resnet50new":
    model = ResNet50NN(
        k=config["k"],
        init_readout_weights=torch.Tensor(w),
        init_readout_bias=torch.Tensor(b),
        freeze_backbone=config["freeze_backbone"])
elif config["model"] == "resnet50":
    model = resnet50(pretrained=False)
else:
    raise ValueError(f'Model "{config["model"]}" does not exists')

## Load pretrained model
if config["base_model"] is not None:
    model.load_state_dict(torch.load(config["base_model"]))



import wandb
# wandb.init()
wandb.init(project='resnet-50-mpl', entity='libeanim', config=config)
    # settings=wandb.Settings(start_method='fork'))
# Delete latent data from memory and trigger garbage collector
# del train_latent
# del val_latent
# import gc; gc.collect()


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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)
    
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"], num_workers=1)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config["batch_size"], num_workers=1)


wandb.watch(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                      weight_decay=config["weight_decay"])

assert config['scheduler'] is None or config['scheduler'] == 'MultiStepLR'

scheduler = None
if config['scheduler'] == "MultiStepLR":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config["milestones"], gamma=config["gamma"])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
model.cuda()

for epoch in range(config["epochs"]):  # loop over the dataset multiple times

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

        # # learning rate decay
        # if scheduler is not None:
        #     scheduler.step()

        # print statistics
        running_loss += loss.item()
        preditions = torch.argmax(outputs, axis=1)
        running_accuracy += (torch.sum(preditions == labels) / outputs.shape[0]).cpu().detach().numpy()
        if i % config["log_steps"] == config["log_steps"] - 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / config["log_steps"]))
            # test_accuracy = get_accuracy(model, val_loader)
            wandb.log({
                "loss": running_loss / config["log_steps"],
                "accuracy": running_accuracy / config["log_steps"],
                "lr": optimizer.param_groups[0]['lr']
            })
            running_loss = 0.0
            running_accuracy = 0
        
    # learning rate decay
    # model.eval()
    # wandb.log({ "test_accuracy": get_accuracy(model, val_loader) })
    # model.train()
    if scheduler is not None:
        scheduler.step()
    
    torch.save(model.state_dict(), f"{RESULT_DIR}/{ config['id'] }_{epoch}.pth")
# model.cpu()
torch.save(model.state_dict(), f"{RESULT_DIR}/{ config['id'] }_final.pth")
print('Finished Training')