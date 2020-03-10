import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import time
import numpy as np
import os.path


# generate sequential number
def generateNum(start, size):
    Numlist = [start]
    num = start
    for i in range(size-1):
        num = num + 1
        Numlist.append(num)

    return Numlist


# number of input channels
num_input_channel = 1

# number of output classes
num_class = 10

# number of Folds
cv_fold = 10

resume_weights = "sample_data/checkpoint1.pth.tar"

cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

# transforming input images into tensors
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, ],
    #                      std=[0.5, ])
])


# Function for training the network
def train_cpu(model, optimizer, train_loader, loss_fun):
    average_time = 0
    total = 0
    correct = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        batch_time = time.time()
        images = Variable(images)
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs, labels)

        if cuda:
            loss.cpu()

        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time
        average_time += batch_time

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % print_every == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f, Batch time: %f'
                  % (epoch + 1,
                     num_epochs,
                     i + 1,
                     len(train_loader),
                     loss.item(),
                     correct / total,
                     average_time / print_every))


# Function for validating the network
def eval_cpu(model, test_loader):
    model.eval()

    total = 0
    correct = 0
    for i, (data, labels) in enumerate(test_loader):
        data, labels = Variable(data), Variable(labels)
        if cuda:
            data, labels = data.cuda(), labels.cuda()

        data = data.squeeze(0)
        labels = labels.squeeze(0)

        outputs = model(data)
        if cuda:
            outputs.cpu()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


# Save the network for the best recognition accuracy on validation set
def save_checkpoint(state, is_best, filename="sample_data/checkpoint.pth.tar"):
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


# The proposed multi-column network
class Multi_Column_Network(nn.Module):
    def __init__(self):
        super(Multi_Column_Network, self).__init__()

        self.layer11 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=3, stride=2, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer21 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=5, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer23 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer31 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=7, stride=1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer32 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer33 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # first column fc layer

        self.fc11 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        self.fc12 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.fc13 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        # second column fc layer

        self.fc21 = nn.Sequential(
            nn.Linear(7200, 3584),
            nn.Dropout(0.5),
            nn.BatchNorm1d(3584),
            nn.ReLU())

        self.fc22 = nn.Sequential(
            nn.Linear(10816, 5120),
            nn.Dropout(0.5),
            nn.BatchNorm1d(5120),
            nn.ReLU())

        self.fc23 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        # third column fc layer

        self.fc31 = nn.Sequential(
            nn.Linear(6272, 2560),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2560),
            nn.ReLU())

        self.fc32 = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        self.fc33 = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        # concatenated features fc layer

        self.fc0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU())

        self.fl1 = nn.Sequential(
            nn.Linear(7168, 3584),
            nn.Dropout(0.5),
            nn.BatchNorm1d(3584),
            nn.ReLU())

        self.fl2 = nn.Sequential(
            nn.Linear(15360, 8192),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8192),
            nn.ReLU())

        self.fl3 = nn.Sequential(
            nn.Linear(11264, 5120),
            nn.Dropout(0.5),
            nn.BatchNorm1d(5120),
            nn.ReLU())

        # final fc layer

        self.fc_final = nn.Sequential(
            nn.Linear(17408, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU())

    def forward(self, x):
        x0 = x.view(-1, self.num_flat_features(x))

        # for first column
        xc11 = self.layer11(x)
        x11 = xc11.view(-1, self.num_flat_features(xc11))

        xc12 = self.layer12(xc11)
        x12 = xc12.view(-1, self.num_flat_features(xc12))

        xc13 = self.layer13(xc12)
        x13 = xc13.view(-1, self.num_flat_features(xc13))

        # for second column
        xc21 = self.layer21(x)
        x21 = xc21.view(-1, self.num_flat_features(xc21))

        xc22 = self.layer22(xc21)
        x22 = xc22.view(-1, self.num_flat_features(xc22))

        xc23 = self.layer23(xc22)
        x23 = xc23.view(-1, self.num_flat_features(xc23))

        # for third column
        xc31 = self.layer31(x)
        x31 = xc31.view(-1, self.num_flat_features(xc31))

        xc32 = self.layer32(xc31)
        x32 = xc32.view(-1, self.num_flat_features(xc32))

        xc33 = self.layer33(xc32)
        x33 = xc33.view(-1, self.num_flat_features(xc33))

        # features from first column

        x11 = self.fc11(x11)
        x12 = self.fc12(x12)
        x13 = self.fc13(x13)

        # features from second column

        x21 = self.fc21(x21)
        x22 = self.fc22(x22)
        x23 = self.fc23(x23)

        # features from third column

        x31 = self.fc31(x31)
        x32 = self.fc32(x32)
        x33 = self.fc33(x33)

        # all concatenated features level wise

        xl1 = torch.cat((x11, x21), 1)
        xl1 = torch.cat((xl1, x31), 1)

        xl2 = torch.cat((x12, x22), 1)
        xl2 = torch.cat((xl2, x32), 1)

        xl3 = torch.cat((x13, x23), 1)
        xl3 = torch.cat((xl3, x33), 1)

        x0 = self.fc0(x0)

        # concatenated features fc layer level wise

        xl1 = self.fl1(xl1)
        xl2 = self.fl2(xl2)
        xl3 = self.fl3(xl3)

        # all features concatenation level wise

        xlz = torch.cat((x0, xl1), 1)
        xlz = torch.cat((xlz, xl2), 1)
        xlz = torch.cat((xlz, xl3), 1)

        # final fc layer

        out = self.fc_final(xlz)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Creating object for the network class
model = Multi_Column_Network()
if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
if cuda:
    criterion.cuda()

# Checking for previously saved model for resume training
if os.path.isfile(resume_weights):
    print("=> loading checkpoint '{}' ...".format(resume_weights))
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))


# Running the model for each fold
for fold in range(cv_fold):
    num_epochs = 100
    batch_size = 500
    learning_rate = 0.001
    print_every = 10
    best_accuracy = torch.FloatTensor([0])
    start_epoch = 0

    model = Binary_Tree()
    if cuda:
        model.cuda()

    # Loading the training data set
    train_set = torchvision.datasets.MNIST(root='/input', train=True, download=True, transform=transform)
    indices = list(range(len(train_set)))
    val_split = int(len(indices) / cv_fold)

    # Creating validation set
    val_idx = generateNum(fold * val_split, val_split)
    train_idx = list(set(indices) - set(val_idx))

    val_sampler = SubsetRandomSampler(val_idx)
    val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=val_sampler,
                                             shuffle=False)
    val_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, sampler=val_sampler, shuffle=False)

    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler,
                                               shuffle=False)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, sampler=train_sampler, shuffle=False)

    # Loading the test data set
    test_set = torchvision.datasets.MNIST(root='/input', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_loader2 = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(test_loader))

    total_step = len(train_loader)

    # Running for each epoch
    for epoch in range(num_epochs):
        print("Fold No: ", fold+1)

        print(learning_rate)

        # Initializing RMSProp optimizer
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        # Variable learning rate
        if learning_rate > 0.00003:
            learning_rate = learning_rate * 0.993

        # Calling the training function
        train_cpu(model, optimizer, train_loader, criterion)
        val_acc = eval_cpu(model, val_loader)
        print('=> Validation set: Accuracy: {:.2f}%'.format(val_acc * 100))

        test_acc = eval_cpu(model, test_loader)
        print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

        acc = torch.FloatTensor([val_acc])

        is_best = bool(acc.numpy() > best_accuracy.numpy())

        best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))

        # Saving the best recognition accuracy on validation set
        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best)
