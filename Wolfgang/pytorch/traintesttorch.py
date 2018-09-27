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

for layers in range(HID_LAY_MIN,HID_LAY_MAX+1,HID_LAY_STEP):
  for nodes in range(NOD_MIN, NOD_MAX+1,NOD_STEP):
    #update hyperparameters
    hyp["layer"] = layers
    hyp["nodes"] = nodes
    hyp["activ"] = "rel"
    hyp["shape"] = "lin"
    #initialize net
    model = CreateNet(hyp)
    loss_fn = nn.MSELoss(size_average=True, reduce=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    #define trainloader
    trainloader = DataLoader(tr_all, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=4)
    #initialize loss lists
    loss_plot_x = []  
    loss_plot_y = []
    loss_plot_y_val = []
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
      loss_plot_x.append(t)  
      loss_plot_y.append(np.sqrt(loss_fn(model(tr_data_torch), tr_labels_torch).detach().numpy()))
      loss_plot_y_val.append(np.sqrt(loss_fn(model(val_data_torch), val_labels_torch).detach().numpy()))
    #make sample predictions with val data and measure time with certain sample size
    for i in range(ANALYSIS_SAMPLE_SIZE):
      t0 = time.time()
      preds = model(val_data_torch)
      t1 = time.time()
      predtime = predtime + t1-t0
    predtime = predtime / ANALYSIS_SAMPLE_SIZE
    loss = min(loss_plot_y_val)
    print (predtime)
    print (loss)
    #save hyperloss (with val data) to dictionary
    hyloss = hyperloss(predtime,loss,INT_LOSS)
    allhloss["l" + str(layers) + "n" + str(nodes)] = hyloss
    #if net in top 10: save hyperl, loss, predt, archit
    if (len(toplist) < 10) or (hyloss < toplist[9][0]):
      toplist.append([hyloss, loss, predtime, layers, nodes, loss_plot_x, loss_plot_y, loss_plot_y_val])
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
    f0.write("\n")
    
#visualize loss for toplist
for i in range(len(toplist)):
  netdata = toplist[i]
  if netdata[6][0] > netdata[6][1] * 2:
      netdata[6][0] = netdata[6][1]
  if netdata[7][0] > netdata[7][1] * 2:
      netdata[7][0] = netdata[7][1]    
  plt.figure(i)  
  plt.title('Loss Function', fontsize=20)
  plt.xlabel('Epochs')
  plt.ylabel('Error')
  plt.plot(netdata[5], netdata[6], label = 'Train Loss')
  plt.plot(netdata[5], netdata[7], label = 'Validation Loss')
  plt.legend()
  plt.savefig("analysis/plots/loss" + str(i) + ".png")

#save dictionary
with open('analysis/allhloss.pkl', 'wb') as f:
  pickle.dump(allhloss, f, pickle.HIGHEST_PROTOCOL)

###to load dictionary again:
#  with open('analysis/allhloss.pkl', 'rb') as f:
#    pickle.load(f)

