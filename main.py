from data import *
from config import *
from datapro import *
from traintest import train,validation
from model import BERT
import torch.nn as nn
import torch.optim as optim


model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
min_loss = 100
for epoch in range(epochs):
    train(model=model, loader=loader, optimizer=optimizer, criterion=criterion,vocab_size=vocab_size, epoch=epoch)
    loss = validation(model=model, loader=val_loader,criterion=criterion,vocab_size=vocab_size)
    print("val loss at epoch {}:{}".format(epoch,loss))
    if loss<min_loss:
        print("save at epoch {}".format(epoch))
        min_loss = loss

