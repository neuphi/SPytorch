def hyperloss(time, loss, intloss):
  a = 1     #some finetuning is possible with the a,b parameters
  b = 1
  c = 1
  if b*(loss-intloss) > 10:
    return a*time + c*np.exp(10)
  else:
    return a*time + c*np.exp(b*(loss-intloss))
