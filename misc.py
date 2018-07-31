from glovar import *
import pickle
from smodels.experiment.databaseObj import Database

def getExpRes(exp):
    print("load result ..")
    database=Database(PATH_DATABASE)
    print("  done")
    return database.getExpResults(analysisIDs = [exp])[0]

def saveData(data):
    print("\nsaving data ..")
    file = open(PATH_DATA + "data.pcl","wb")
    pickle.dump(len(data), file)
    for item in data:
        pickle.dump(item, file)
    file.close
    print("  done")
    
def loadData():
    print("\nloading data ..")
    file = open(PATH_DATA + "data.pcl","rb")
    lgth = pickle.load(file)
    data = []
    for i in range(lgth):
        data.append(pickle.load(file))
    file.close()
    print("  done")
    return data

def Hash ( A ):
    return int(A[0]*10000.+A[1])
