import numpy as np

def hyperloss(time, loss, intloss):
  a = 1000     #some finetuning is possible with the a,b parameters
  b = 5
  c = 1
  if b*(loss-intloss) > 10:
    return a*time + c*np.exp(10)
  else:
    return a*time + c*np.exp(b*(loss-intloss))
