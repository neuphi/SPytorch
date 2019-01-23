# spytorch

= Philipp Neuhuber =

== Project ==

 * Diploma thesis, started dec 1st, 2018
 
 * goal: expand the .getUpperLimitFor(*args) functionality of expResultObj by providing an MLP network prediction option
         for each topology and experimental result
 * goal: implement a few ATLAS analyses to test the networks

== TODO ==
  
 * figure out where and how to save weights and shape (.h5 file - multiple topologies per file?)
 * find an already existing method to efficiently read database Tx<Name>.txt files
 * improve search algorithm runtime (figure out why cuda is 2-3x slower than cpu)
 * test out various methods of weight initialization (JKU)
 * expand the search algorithm to fully supply all topologies and results
 * implement the first prototype into SModelS

== Done ==

 * expand existing grid search algorithm to support
 * split gridsearch into seperate algorithms for searchparameter and netparameter
