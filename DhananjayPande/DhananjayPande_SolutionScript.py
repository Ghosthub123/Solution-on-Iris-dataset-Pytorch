import numpy as np
import torch
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
cols = data.feature_names
myData = data.data
df = pd.DataFrame(myData,columns=cols)
target = data.target
df['target'] = target

from torch.utils.data import TensorDataset , DataLoader 

data = df.drop('target',axis = 1).values
labels = df.target.values

iris = TensorDataset(torch.FloatTensor(data),torch.LongTensor(labels))
iris_loader = DataLoader(iris,batch_size = 50,shuffle = True)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,in_features = 4 , h1= 8 , h2 = 9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x

torch.manual_seed(32)
model = Model()

from sklearn.model_selection import train_test_split

X = df.drop("target",axis = 1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters() , lr = 0.01)

train_losses = []
test_losses = []
epochs = 100
weights = []
biases=[]
train_accuracy = []
test_accuracy = []
incr = 0
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    y_pred = model.forward(X_train)
    
    # CALC ERROR
    
    loss_train = criterion(y_pred,y_train)
    
    train_losses.append(loss_train)
    weights.append(model.out.weight[0][0].item())
    biases.append(model.out.bias[0].item())
    predicted = torch.max(y_pred.data, 1)[1]
    epoch_corr = (predicted == y_train).sum()
    trn_corr+=epoch_corr
    accuracy = (trn_corr.item()*100)/(len(y_train))
    train_accuracy.append(accuracy)
        
    #Backpropagation
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    with torch.no_grad():
        y_val = model.forward(X_test)
        loss_test = criterion(y_val,y_test)
        predicted_test = torch.max(y_val.data, 1)[1]
        epoch_corr_test = (predicted_test == y_test).sum()
        tst_corr+=epoch_corr_test
        accuracy_test = (tst_corr.item()*100)/(len(y_test))
        test_accuracy.append(accuracy_test)
        test_losses.append(loss_test)
    if i%10==0:
        [w,b] = model.out.parameters()
        print(f"Epochs : {i}, Training loss : {loss_train} , weight : {w[0][0].item():3f} , bias : {b[0].item():3f} , Training accuracy: {trn_corr.item()*100/(len(y_train)):7.3f}%') \n  Testing loss : {loss_test} , Testing accuracy : {tst_corr.item()*100/(len(y_test)):7.3f}%")   
       
print(" \n K0 value is  = ",weights[-1], " k1 value is = ",biases[-1])

def changeplot():
    fig,axes = plt.subplots(1,2,figsize = (10,12),subplot_kw={"xlabel":"Epochs"})
    axes[0].plot(range(epochs),weights)
    axes[0].set_title("Weight vs epochs")
    axes[1].plot(range(epochs),biases)
    axes[1].set_title("Bias vs epochs")
    return plt.show()
 
def lossplot():
    plt.plot(train_losses,test_losses)
    plt.xlabel("train")
    plt.ylabel("test")
    return plt.show()
    
def accuracyplot():
    plt.scatter(train_accuracy,test_accuracy)
    plt.xlabel("train")
    plt.ylabel("test")
    return plt.show()
    
def lossfunctionplot():
    plt.plot(range(epochs),train_losses)
    plt.plot(range(epochs),test_losses)
    plt.legend(labels = ['train','test'])
    plt.xlabel("Epochs")
    return plt.show()
    
print("\n Changes in k0 and k1 with each epochs")
changeplot()
print(" \n Training loss vs Test loss")
lossplot()
print(" \n Training accuracy vs test accuracy")
accuracyplot()
print(" \n Loss function plot")
lossfunctionplot()