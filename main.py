import pandas as pd
import json
import matplotlib.pyplot as plt
from random import shuffle
import torch

from model import TimeCNN
from data_preprocessing import create_data_dict

with open('config.json',) as file : 
    paths = json.load(file)

""" We are intrested in traffic volume prediction in Austin City"""

def split_ts(seq, horizon=24*30, n_steps=24*30*3):
    """ we want to decompose each time series into multiple observations
    using a sliding window:
    
    this function take in arguments a traffic Time Series for the couple (l,d)
    and applies a sliding window of length n_steps to generates samples having this 
    length and their labels (to be predicted) whose size is horizon
    """
    #for the Min-Max normalization X-min(seq)/max(seq)-min(seq)
    max_seq = max(seq)
    min_seq = min(seq)
    seq_norm = max_seq - min_seq
    xlist, ylist = [], []
    for i in range(len(seq)//horizon):
        end = i*horizon + n_steps
        if end+horizon > len(seq)-1:
            break
        xx = (seq[i*horizon:end] - min_seq)/seq_norm
        xlist.append(torch.tensor(xx, dtype=torch.float32))
        yy = (seq[end:(end+horizon)] - min_seq)/seq_norm
        ylist.append(torch.tensor(yy, dtype=torch.float32))
    print("number of samples %d and sample size %d (%d months)" %(len(xlist), len(xlist[0]),n_steps/(24*30)))
    return(xlist, ylist)


def train_test_set(xlist, ylist):
    """ this functions splits the samples and labels datasets xlist and ylist
    (given by the function split_ts) into a training set and a test set
    """
    data_size=len(xlist)
    test_size=int(data_size*0.2) #20% of the dataset
    #training set
    X_train  = xlist[:data_size-test_size]
    Y_train = ylist[:data_size-test_size]
    #test set
    X_test = xlist[data_size-test_size:]
    Y_test = ylist[data_size-test_size:]
    return(X_train, Y_train, X_test, Y_test)

def train_validate_model(mod,
                         seq,
                         num_ep=60,
                         horizon=24*30,
                         n_steps=24*30*3):
    """inputs are the model mod, the Time Series sequence and the number of epochs
    """
    #trainingthe model
    xlist,ylist = split_ts(seq,horizon, n_steps)
    X_train,Y_train,X_test,Y_test = train_test_set(xlist, ylist)
    idxtr = list(range(len(X_train)))
    #loss and optimizer
    loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(mod.parameters(), lr=0.0005)
    loss_val_train = []
    loss_val_test = []
    for ep in range(num_ep):
        shuffle(idxtr)
        ep_loss = 0
        test_loss = 0
        mod.train()
        for j in idxtr:
            opt.zero_grad()
            #forward pass
            haty = mod(X_train[j].view(1,1,-1))
            # print("pred %f" % (haty.item()*vnorm))
            lo = loss(haty,Y_train[j].view(1,-1))
            #backward pass
            lo.backward()
            #optimization
            opt.step()
            ep_loss += lo.item()
        loss_val_train.append(ep_loss)
        #model evaluation
        mod.eval()
        for i in range(len(X_test)):    
            haty = mod(X_test[i].view(1,1,-1))
            test_loss+= loss(haty, Y_test[i].view(1,-1)).item()
        loss_val_test.append(test_loss)
        if ep%50 == 0:
            print("epoch %d training loss %1.9f test loss %1.9f" % (ep, ep_loss, test_loss))
    #test_loss is given for the selected model (last epoch)
    epochs = [i for i in range(num_ep)]
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_val_train, label='training loss')
    ax.plot(epochs, loss_val_test, label='test loss')
    ax.legend()
    plt.show()
    return ep_loss,test_loss

    
"""TRAINING AND EVALUATION OF THE MODEL FOR EACH (LOCATION,DIRECTION)
"""
data = pd.read_csv(paths["data"])
data_dict = create_data_dict(data)

results = pd.DataFrame( columns = ["couple", "training_loss", "test_loss"])
num_ep = 500
horizon = 24*30
n_steps = 24*30*5
for l,d in list(data_dict.keys())[5:]:
    couple = (l,d)
    seq = data_dict[(l,d)] #volume sequence for (l,d) location, direction
    xlist,ylist = split_ts(seq, horizon, n_steps)
    print("couple:",(l,d))
    print("number of samples in the dataset:", len(xlist))
    mod = TimeCNN()
    train_loss, test_loss = train_validate_model(mod, seq, num_ep, horizon, n_steps)
    print("train_loss, test_loss =", train_loss, test_loss, "\n")
    results.loc[len(results)] = [couple, train_loss, test_loss]
    del(mod)

