# spytorch

= Philipp Neuhuber =

== Project ==

 * Diploma thesis, started dec 1st, 2018
 
 * goal: expand the .getUpperLimitFor(*args) functionality of expResultObj by providing an MLP network prediction option
 * goal: implement a few ATLAS analyses to test the networks

== TODO ==
  
 * store nn folder in smodels or utils?
 * if neural net found, automatically use, else interpolate?
 * expand nn to 4(+4) input parameter for both branches?
 * improve search algorithm runtime (figure out why cuda is 2-3x slower than cpu)
 * test out various methods of weight initialization (JKU) (autoencoder)

== Done ==

 * implement the first prototype into SModelS
 * figure out best way so save models -> .pth
 * expand existing grid search algorithm to support multiple topologies and compare results
 * split gridsearch into seperate algorithms for searchparameter and netparameter
 * update to torch v0.4
