import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device, "detected. Use it for learning.")

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

######### implement here 1 ##########
# Hyperparameters setting
lr = 0.001
epochs = 5
batch_size = 100
#####################################

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

######### implement here 2 ##########
# Building a MLP model
layer1 = nn.Linear(784, 512, bias=True)
layer2 = nn.Linear(512, 512, bias=True)
layer3 = nn.Linear(512, 512, bias=True)
layer4 = nn.Linear(512, 10, bias=True)

nn.init.xavier_uniform_(layer1.weight)
nn.init.xavier_uniform_(layer2.weight)
nn.init.xavier_uniform_(layer3.weight)
nn.init.xavier_uniform_(layer4.weight)

relu = nn.ReLU()

model = nn.Sequential(
    layer1, relu,
    layer2, relu,
    layer3, relu,
    layer4
).to(device)
#####################################

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_batch = len(data_loader)
model.train()

for epoch in range(epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        ######### implement here 3 ##########
        # Run training by gradient descent
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        #####################################

        avg_cost += cost / total_batch
    print("Epoch:", epoch + 1, ', cost = {:.5f}'.format(avg_cost))

print("Done Learning")

r = int(input("Choose a random integer between 0 and {:} : ".format(len(mnist_test) - 1)))
X_single_data = mnist_test.data[r : r+1].view(-1, 28 * 28).float().to(device)
Y_single_data = mnist_test.targets[r : r+1].to(device)

with torch.no_grad():
    ######### implement here 4 ##########
    # Predict a single data with trained model
    print("Label:", Y_single_data.item())
    prediction = model(X_single_data)
    print("Prediction:", torch.argmax(prediction, 1).item())
    #####################################

    plt.imshow(mnist_test.data[r : r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

X_test = mnist_test.data.view(-1, 28 * 28).float().to(device)
Y_test = mnist_test.targets.to(device)

with torch.no_grad():
    ######### implement here 5 ##########
    # Predict whole dataset with trained model
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy:", accuracy.item())
    #####################################