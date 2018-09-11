####################### IMPORT STUFF ############################

from misc import *
import matplotlib.pyplot as plt
from hyperloss import *
#import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as Dataset
import time
from initnet import *

######################## LOAD DATA ################################

tr_data,tr_labels,val_data,val_labels = loadData()

SAMPLE_NMBR    = len(tr_data)
SAMPLE_NMBR_VAL= len(val_data)

X={} #create dictionary
for d,l in zip (tr_data,tr_labels):
    X[Hash(d)]=l

###################### MODIFY DATA FOR NN #########################

tr_data_torch    = torch.zeros(SAMPLE_NMBR, DIM_IN)
tr_labels_torch  = torch.zeros(SAMPLE_NMBR, DIM_OUT)
tr_all           = torch.zeros(SAMPLE_NMBR, DIM_IN + DIM_OUT)
val_data_torch   = torch.zeros(SAMPLE_NMBR_VAL, DIM_IN)
val_labels_torch = torch.zeros(SAMPLE_NMBR_VAL, DIM_OUT)

for i in range(SAMPLE_NMBR):
  tr_data_torch[i]   = torch.from_numpy(tr_data[i])
  tr_labels_torch[i] = tr_labels[i]
  tr_all[i][0] = tr_data_torch[i][0]
  tr_all[i][1] = tr_data_torch[i][1]
  tr_all[i][2] = tr_labels_torch[i][0]

for i in range(SAMPLE_NMBR_VAL):
  val_data_torch[i]   = torch.from_numpy(val_data[i])
  val_labels_torch[i] = val_labels[i]

##################### SPLIT DATA #################################

#if SPLIT_CHOOSE = 1, split Data into SPLIT
#val data is already simulated



##################### TRAINING ###################################

#create dictionary and toplist 
allhloss = {}
toplist = []

#add outer loops: lossfunc, activation, minibatch, slope
#optimizer muss noch!
#lossf = ['mse','l1','nll']
#activf = ['linear','relu','tanh','sigmoid','linz']
#minibatchsize = [1,4,16,64,256]
#slope = [lin, ramp, trap]

for layers in range(HID_LAY_MIN,HID_LAY_MAX,int(round((HID_LAY_MAX-HID_LAY_MIN)/2))):
  for nodes in range(NOD_MIN, NOD_MAX, int(round((NOD_MAX-NOD_MIN)/2))):
    #initialize net
    hyp["layer"] = layers
    hyp["nodes"] = nodes
    hyp["activ"] = "rel"
    hyp["shape"] = "lin"
    model = CreateNet(hyp)
    loss_fn = nn.MSELoss(size_average=True, reduce=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    #define trainloader
    trainloader = DataLoader(tr_all, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    #loop over epochs
    for t in range(EPOCH_NUM):
      #loop over trainloader
      for i, data in enumerate(trainloader):
        inputs    = torch.zeros(torch.numel(data[:,0]), DIM_IN)
        labels    = torch.zeros(torch.numel(data[:,0]), DIM_OUT)
        for j in range(torch.numel(data[:,0])):
          inputs[j][0] = data[j][0]
          inputs[j][1] = data[j][1]
          labels[j]    = data[j][2]
        labels_pred = model(inputs)
        loss = loss_fn(labels_pred, labels)
        #make one step to optimize weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      #fill loss lists with data
      #loss_plot_x.append(t)  
      #loss_plot_y.append(loss)
    #make sample predictions with val data and measure time
    t0 = time.time()
    preds = model(val_data_torch)
    t1 = time.time()
    predtime = t1-t0
    loss = loss_fn(preds, val_labels_torch)
    print (predtime)
    print (loss)
    print (INT_LOSS_SQ)
    #save hyperloss (with val data) to dictionary
    hyloss = hyperloss(predtime,loss.detach().numpy(),INT_LOSS_SQ)
    allhloss["l" + str(layers) + "n" + str(nodes)] = hyloss
    #if net in top 10: save hyperl, loss, predt, archit
    if (len(toplist) < 10) or (hyloss < toplist[9][0]):
      toplist.append([hyloss, loss, time, layers, nodes])
      toplist.sort(key = lambda dat: dat[0])
      if len(toplist) > 10:
        toplist.pop()
    
#save toplist
with open('analysis/toplist.txt', 'w') as f0:
  #f0.write("All hyperloss, loss and prediction times eruated with a " + str(SAMPLE_NMBR_VAL) " samples containing validation data set.\n")
  for netdata in toplist:
    f0.write("Hyperloss: " + str(netdata[0]) + "\n")
    f0.write("Loss: " + str(netdata[1]) + "\n")
    f0.write("Prediction time: " + str(netdata[2]) + "\n")
    f0.write("Layers: " + str(netdata[3]) + "\n")
    f0.write("Nodes: " + str(netdata[4]) + "\n")
    f0.write("\n ")

#save dictionary
with open('analysis/allhloss.pkl', 'wb') as f:
  pickle.dump(allhloss, f, pickle.HIGHEST_PROTOCOL)

###to load dictionary again:
#  with open('analysis/allhloss.pkl', 'rb') as f:
#    pickle.load(f)

