# spytorch

= Philipp Neuhuber =

== Project ==

 * Diploma thesis, started dec 1st, 2018
 
 * goal: expand the .getUpperLimitFor(*args) functionality of expResultObj by providing an MLP network prediction option
 * goal: implement a few ATLAS analyses to test the networks

== TODO ==

 * ATLAS analyses:
 
  ATLAS SUSY 2016 32 - search for heavy charged long-lived particles
					 no specific topology
  ATLAS SUSY 2017 01 - Can't access HEP Data, Permission denied
					 topologies dont exist yet eg [[W,h]] -> [[q,q],[b,b]]
  ATLAS SUSY 2016 30 - too many dof for both topologies

  ATLAS SUSY 2016 31 - unusable (I suppose), same as 2016 32

  ATLAS SUSY 2017 02 - T1bbbb
					 no proper UL file, create 2 entries for low energy and high energy search or only pick high?
  ATLAS SUSY 2016 21 - too many dof
					 baryon-number-violating lambda??
  ATLAS SUSY 2016 22 - baryon-number-violating lambda''??

  ATLAS SUSY 2016 24 - Fig 1a: TChipChimSlepSnu [[[L-],[nu]],[[nu],[L+]]]+
							 				  [[[L+],[nu]],[[nu],[L-]]]+
							 				  [[[L+],[nu]],[[L-],[nu]]]+
							 				  [[[nu],[L+]],[[nu],[L-]]]		?
					 Fig 1a: oppositely charged charginos do not add additional dof in SMS
					 Fig 1b: too many dof (3)?
					 Fig 1c/1d: no constraint -> too many dof (3)? 
								also no distinguishment between c and d?
  ATLAS SUSY 2016 27 - Fig 1a/1b/1c/1d: no constraints -> too many dof (3)?

  ATLAS SUSY 2016 25 - 120 tables for 2 topolgies -> what is going on here?

  ATLAS SUSY 2016 07 - 426 data tables

  //ATLAS SUSY 2016 06 - no longer in ATLAS SUSY list??
					   weird topologies
  
 * how to find and properly import smodels from within utils
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
