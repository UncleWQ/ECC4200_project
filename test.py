import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform_train = transforms.Compose(
    [
     transforms.RandomCrop(32,padding = 4),
     transforms.RandomHorizontalFlip(p = 0.5),    #随机水平翻转
     transforms.ToTensor(),   #转换为张量
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #标准化
    ])
 
transform = transforms.Compose(
    [ 
      # transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

train_set_new = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_set_new = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_size_new = len(train_set_new)
print(train_size_new)
test_size_new = len(test_set_new)
print(test_size_new)

class ResidualBlock_new(nn.Module):
    def __init__(self, input_channels, num_channels, strides = 1):
        super(ResidualBlock_new, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.downsample = None
        if strides != 1 or input_channels != num_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(num_channels)
            )

    def forward(self, X):
        identity = X
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(X)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet18_IMprove(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_IMprove, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock_new(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock_new(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet18_IMprove()
print(model)

# Hyperparameters
epochs = 10
batch_size = 256
learning_rate = 0.01

# Set up optimizer
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Build data loaders
train_loader_new = torch.utils.data.DataLoader(train_set_new, batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader_new = torch.utils.data.DataLoader(test_set_new, batch_size = batch_size, shuffle = False, num_workers = 0)

data_loaders_new = {"train": train_loader_new, "test": test_loader_new}
dataset_sizes_new = {"train": train_size_new, "test": test_size_new}

def eval_on_test_set(model):
    model.eval()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model.to(device)
    running_accuracy = 0
    loss = 0

    for data in test_loader_new:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        running_accuracy += correct

    total_loss = loss / test_size_new
    total_accuracy = running_accuracy / test_size_new
    print('Evaluation on test set: loss={:.3f} \t accuracy={:.2f}%'.format(total_loss, total_accuracy * 100))
    model.train()
    return total_loss, total_accuracy


def train_for_one_epoch(model):
    model.train()
    # Set up device
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name()
    print('Using device '+ name + ' to train the model.')
    model.to(device)

    # set the running quatities to zero at the beginning of the epoch
    running_loss = 0
    running_accuracy = 0

    for data in train_loader_new:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        # compute running statistics
        running_loss += batch_loss.item()
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        running_accuracy += correct


    # Compute stats for the full training set
    total_loss = running_loss / train_size_new
    total_accuracy = running_accuracy / train_size_new

    return total_loss, total_accuracy
# start training
metrics = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
for epoch in range(epochs):
  start=time.time()
  train_loss_epoch, train_acc_epoch = train_for_one_epoch(model)
  elapsed = (time.time()-start) / 60
  print('Training epoch={} \t cost_time={:.2f} min \t loss={:.3f} \t accuracy={:.2f}%'.format(epoch, elapsed, train_loss_epoch, train_acc_epoch * 100))
  test_loss_epoch, test_acc_epoch = eval_on_test_set(model)
  metrics['train_loss'].append(train_loss_epoch)
  metrics['train_acc'].append(train_acc_epoch)
  metrics['test_loss'].append(test_loss_epoch)
  metrics['test_acc'].append(test_acc_epoch)

# save your trained model for the following question
torch.save(model.state_dict(), './model_resnet18_Improve.pt')


train_loss = metrics['train_loss']
test_loss = metrics["test_loss"]
train_accuracy = metrics['train_acc']
test_accuracy = metrics['test_acc']
epochs = range(1, len(train_loss) + 1)

# 创建一个包含两个子图的画布
fig, axs = plt.subplots(2, figsize=(12, 12))

# 绘制训练集和测试集的损失曲线
axs[0].plot(epochs, train_loss, 'b', label='Training loss')
axs[0].plot(epochs, test_loss, 'r', label='Testing loss')
axs[0].set_title('Loss Curves')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# 绘制训练集和测试集的准确率曲线
axs[1].plot(epochs, train_accuracy, 'b', label='Training accuracy')
axs[1].plot(epochs, test_accuracy, 'r', label='Testing accuracy')
axs[1].set_title('Accuracy Curves')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()
