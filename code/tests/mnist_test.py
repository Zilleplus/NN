import torch
import torchvision.datasets as ds # type: ignore
import torchvision.transforms as transforms # type: ignore
from torch import nn

mnist_train_data = ds.MNIST(download=True,
                            root='MNIST/processed/training.pt',
                            transform=transforms.ToTensor())
mnist_test_data = ds.MNIST(download=True,
                           root='MNIST/processed/test.pt',
                           transform=transforms.ToTensor())

batch_size = 128
train_data_loader = torch.utils.data.DataLoader(dataset=mnist_train_data,
                                                batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(dataset=mnist_test_data,
                                               batch_size=batch_size)

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.layer1 = nn.Sequential(
#             nn.Linear(in_features=28*28, out_features=32),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Linear(in_features=32, out_features=10),
#             nn.Softmax())
# 
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x

# model = NeuralNetwork()
# print(model)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # model.train()
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X, y
        y_one_hot = nn.functional.one_hot(y, num_classes=10).type(torch.FloatTensor)

        # # Compute prediction error
        # pred = model(X)
        # loss = loss_fn(pred, y_one_hot)

        # # Backpropagation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for batch, (X, y) in enumerate(dataloader):
#             X, y = X, y
# 
#             pred = model(X)
# 
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%\
#           , Avg loss: {test_loss:>8f} \n")


epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    # train(train_data_loader, model, loss_fn, optimizer)
    # test(test_data_loader, model, loss_fn)
print("Done!")