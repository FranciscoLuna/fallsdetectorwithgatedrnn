# -*- coding: utf-8 -*-

import os
import numpy as np
import math
import random

from copy import deepcopy

# Data model

class FallDatabase():
    
    
    ## Repetitions and temporal samples per activity
    
    activityDataByType = {1: [1, 100, 20000], 2: [5, 25, 5000], 3: [5, 12, 2400], 4: [5, 15, 3000]}
    
    activitiesClasification = {"D01": 1,"D02": 1,"D03": 1,
                               "D04": 1,"D05": 2,"D06": 2,
                               "D07": 3,"D08": 3,"D09": 3,
                               "D10": 3,"D11": 3,"D12": 3,
                               "D13": 3,"D14": 3,"D15": 3,
                               "D16": 3,"D17": 2,"D18": 3,
                               "D19": 3,
                               
                               "F01": 4,"F02": 4,"F03": 4,
                               "F04": 4,"F05": 4,"F06": 4,
                               "F07": 4,"F08": 4,"F09": 4,
                               "F10": 4,"F11": 4,"F12": 4,
                               "F13": 4,"F14": 4,"F15": 4
                              }  
    
    
    ## Subjects classified by set of activities carried out
    # Note: It is not entirely real, some type-2 subjects did not perform some of the tasks contemplated
    subjectType = {
            "SA01":1,"SA02":1,"SA03":1,"SA04":1,"SA05":1,
            "SA06":1,"SA07":1,"SA08":1,"SA09":1,"SA10":1,
            "SA11":1,"SA12":1,"SA13":1,"SA14":1,"SA15":1,
            "SA16":1,"SA17":1,"SA18":1,"SA19":1,"SA20":1,
            "SA21":1,"SA22":1,"SA23":1,
            
            "SE01":2,"SE02":2,"SE03":2,"SE04":2,"SE05":2,
            "SE06":1,"SE07":2,"SE08":2,"SE09":2,"SE10":2,
            "SE11":2,"SE12":2,"SE13":2,"SE14":2,"SE15":2
            
            }
    
    subjectTypeActivities = {1:["D01","D02","D03",
                               "D04","D05","D06",
                               "D07","D08","D09",
                               "D10","D11","D12",
                               "D13","D14","D15",
                               "D16","D17","D18",
                               "D19",
                               
                               "F01","F02","F03",
                               "F04","F05","F06",
                               "F07","F08","F09",
                               "F10","F11","F12",
                               "F13","F14","F15"
                              ]
                              ,
                             2:
                              ["D01","D02","D03",
                               "D04","D05",
                               "D07","D08","D09",
                               "D10","D11","D12",
                                     "D14","D15",
                               "D16","D17"
                              ]
                          }

    
    ##
    # This method search for each folder (users) and analyze every file 
    # (activity and repetition) to create and return a 3-level dictionary
    #
    # users -> activities -> repetitions
    # 
    # 'filesWithLabels' indicates if the path provided contains files
    # with the data recorded with sensors or the classification labels for each
    # recording.
    #
    #
    # The 'repetition' pairs contains as values:
    #
    # - If files contains the data recorded by sensors: 
    #   A (n_samples x 9) ndarray. Each column refers to following:
    #
    #     1. Acceleration - x axis (sensor ADXL345)
    #     2. Acceleration - y axis (sensor ADXL345)
    #     3. Acceleration - z axis (sensor ADXL345)
    #     4. Rotation - x axis  (sensor ITG3200)
    #     5. Rotation - y axis  (sensor ITG3200)
    #     6. Rotation - z axis  (sensor ITG3200)
    #     7. Acceleration - x axis (sensor MMA8451Q)
    #     8. Acceleration - y axis (sensor MMA8451Q)
    #     9. Acceleration - z axis (sensor MMA8451Q)
    #
    # - If files contains the clasification labels:
    #   A (n_samples) ndarray containing the labels
    #
    #
    def importData(self, path, filesWithLabels = False):
        
        data = dict()
        
        dirList = os.listdir(path)
        listSice = len(dirList)
        count = 0;
        for directory in os.listdir(path):
            print("Completed: " + str(count/listSice) + "%")
            print("Processing folder no: " + str(count+1))
            fullElementRoute = path + "/" + directory
            
            if os.path.isdir(fullElementRoute):
                
                data[directory] = dict() 
                
                for file in os.listdir(fullElementRoute):
                    
                    fullFileRoute = fullElementRoute + "/" + file
                    if not os.path.isdir(fullFileRoute) and ".txt" in file:
                        
                        activityName, subject, repetition = file.split(".")[0].split("_")
                        
                        
                        if(filesWithLabels):
                            currentFile = np.genfromtxt(fullFileRoute, delimiter= None, dtype ='int', skip_footer=1)
                        else:
                            currentFile = np.genfromtxt(fullFileRoute, delimiter= ",", dtype ='int', converters={8: lambda x: int(x[:-1])}, skip_footer=1)
                        

                        if not activityName in data[directory]:
                            data[directory][activityName] = dict()
                        
                        data[directory][activityName][repetition] = currentFile
            
            count = count +1 
        
        return data
    #
    ##
    
    ##
    # This method recives the data structured with the same shape returned by
    # the 'importData' function. 
    # It divides the data in two sets: for train and for test.
    # For that, all users are firstly divided in two groups: users who carried
    # out all activities and not. Secondly, both two sets are randomly conformed
    # but the test set contains approximately 50 % of each group.
    #
    # Both train and test sets are now structured as list of list 
    # (number_of_users x number_of_records), containing: 
    # - Data: (number_of_samples x 9) ndarray containing the sensor recording
    # - Labels: (n_samples) ndarray containing the labels
    #
    # The method returns both train and test sets divided in values and labels.
    # The method also return the distribution obtained randomly to make possible
    # the analysis of results per each group and user individually
    # 10% Test approximately by default (4 subjects)
    #
    def structuredData(self, dataValues, dataClasif, nTestSubjects = 4, softData=False):
        
        trainData = list()
        testData = list()
        
        trainClasification = list()
        testClasification = list()
        
        dataOrganization = {'train': [],
                            'test': []}
        
        subjectType1List = [i for i in self.subjectType.keys() if self.subjectType[i] == 1] 
        subjectType2List = [i for i in self.subjectType.keys() if self.subjectType[i] == 2]
        
        nType1ForTest = math.ceil(nTestSubjects/2)
        nType2ForTest = math.floor(nTestSubjects/2)
        
        subjectsType1SelectedForTest = random.sample(subjectType1List, nType1ForTest)
        subjectsType2SelectedForTest = random.sample(subjectType2List, nType2ForTest)
        
        
        # The 3-level dictionary used was created following an alphabetical order.
        # Thus, when we use keys() and values() methods, that returns the items
        # in insertion order, is equivalent to alphabetical order.
        for subject in dataValues.keys():
            count = 0
            for activity in dataValues[subject].keys():
                count = count + 1;
                if subject in subjectsType1SelectedForTest + subjectsType2SelectedForTest:
                    testData = testData + list(dataValues[subject][activity].values())
                    testClasification = testClasification + list(dataClasif[subject][activity].values())
                    for repetition in dataValues[subject][activity].keys():
                    #if not subject in dataOrganization['test']:
                        dataOrganization['test'].append((subject, activity, repetition))
                    
                else:
                    trainData = trainData + list(dataValues[subject][activity].values())
                    trainClasification = trainClasification + list(dataClasif[subject][activity].values())
                    for repetition in dataValues[subject][activity].keys():
                    # if not subject in dataOrganization['train']:
                        dataOrganization['train'].append((subject, activity, repetition))
        
        return trainData, trainClasification, testData, testClasification, dataOrganization 
    #
    ##
    
    ##
    #
    # This method cut up train and test sets provided in blocks of 'windowSize' length sliding by a 'stride'.
    #
    def assignWindowLabel(self, dataTrValues, dataTrLabels, dataTestValues, dataTestLabels, dataOrganization, windowSize, stride, randomOrder = False):
        
        dataTrWinValues, dataTrWinLabel, dataTrWinOrganization = self.getWindowLabelClasification(dataTrValues, dataTrLabels, dataOrganization['train'], windowSize, stride, randomOrder)
        dataTestWinValues, dataTestWinLabel, dataTestWinOrganization = self.getWindowLabelClasification(dataTestValues, dataTestLabels, dataOrganization['test'], windowSize, stride, randomOrder)
        
        dataWinOrganization = {'train': dataTrWinOrganization, 'test': dataTestWinOrganization}
        
        return np.array(dataTrWinValues), np.array(dataTrWinLabel), np.array(dataTestWinValues), np.array(dataTestWinLabel), dataWinOrganization
    #
    ##
    
    ##
    # This method cut up the data provided in blocks of 'windowSize' length sliding by a 'stride'.
    # Each block is classified using the same criteria as in 
    # TODO: 'randomOrder' alternative is not implemented yet
    # The 'randomOrder' parameter indicates if the cutted up data is returned in a random order.
    #
    def getWindowLabelClasification(self, dataValues, dataLabels, dataOrganization, windowSize, stride, randomOrder=False):
        
        dataWindowValues = list()
        dataWindowLabel = list()
        dataWindowOrganization = list()
        
        for i in range(len(dataLabels)):
            
            sample = dataLabels[i]
            strideAcum = 0
            
            while(strideAcum + windowSize <= len(sample)):
                
                currentWindowSample = sample[strideAcum: (strideAcum + windowSize)]
                
                if np.count_nonzero(currentWindowSample == 2) >= (windowSize*0.1):
                    currentWindowLabel = 2
                elif np.count_nonzero(currentWindowSample == 1) >= (windowSize*0.5):
                    currentWindowLabel = 1
                else:
                    currentWindowLabel = 0
                
                dataWindowLabel.append(currentWindowLabel)
                dataWindowValues.append(dataValues[i][strideAcum: (strideAcum + windowSize)])
                dataWindowOrganization.append(dataOrganization[i])
                strideAcum = strideAcum + stride
                
        
        return dataWindowValues, dataWindowLabel, dataWindowOrganization
    #
    ##
    
    
    # Methods to apply a simple mean filter to eliminate outliers
    
    def softData(self, data, windowSize=10):

        resultsData = deepcopy(data)
        count = 1
        for measureSet in resultsData:
            self.softMeasures(measureSet, windowSize)
            if count%50 == 0:
                print('Measure set n {} processed'.format(count))
            count = count +1
        return resultsData

    def softMeasures(self, measureSet, windowSize=10):

        wC = windowSize//2

        for i in range(0,len(measureSet) - windowSize):

            mean_m = np.mean(measureSet[i:i+windowSize], axis=0)
            std_m = np.std(measureSet[i:i+windowSize], axis=0)

            thr_pos_inf = mean_m + std_m*(2/3)
            thr_neg_inf = mean_m - std_m*(2/3)

            thr_pos_sup = mean_m + std_m*(4/3)
            thr_neg_sup = mean_m - std_m*(4/3)

#            print(measureSet[wC+i])
#            print(thr_pos_inf)
#            print(thr_neg_inf)
#            print(thr_pos_sup)
#            print(thr_neg_sup)
#            print(measureSet[wC+i-1])

            measureSet[wC+i] = np.where( ((measureSet[wC+i] > thr_pos_inf) & (measureSet[wC+i] < thr_pos_sup)) \
                                          | ( (measureSet[wC+i] < thr_neg_inf) & (measureSet[wC+i] > thr_neg_sup)), 
                                            measureSet[wC+i-1], measureSet[wC+i])
##
#
# This method use the FallData class to structure the dataset (originally separated in files) 
# correctly in train and test sets.
# Additionaly, it creates two files (values and labels) in numpy format for a 
# better performance loading the information in the future.
#
# TODO: if it worth, implement to save soft data version. If not, delete this comment
#
def prepareDataset(dataFilesRootpath='./SisFallData/'):
    
    valuesFileName = 'SisFallDataNumpy'
    labelsFileName = 'SisFallLabelsNumpy'
    fileExtensionName = ''
    fileFormat = '.npy'
    
    valuesCompleteName = valuesFileName + fileExtensionName + fileFormat
    labelsCompleteName = labelsFileName + fileExtensionName + fileFormat
    
    sisFallDatabaseInstance = FallDatabase()
    
    if os.path.isfile(dataFilesRootpath + valuesCompleteName) \
       and os.path.isfile(dataFilesRootpath + labelsCompleteName):
        print("Data found in numpy format. Loading them...")
        dataV = np.load(dataFilesRootpath + valuesCompleteName)[()]
        dataL = np.load(dataFilesRootpath + labelsCompleteName)[()]
        print("Done!")
    else:
        print("Data not found in numpy format. Loading raw data ...")
        dataV = sisFallDatabaseInstance.importData(dataFilesRootpath + 'SisFall_dataset')
        dataL = sisFallDatabaseInstance.importData(dataFilesRootpath + 'SisFall_temporally_annotated', filesWithLabels=True)
        print("Loading complete! Saving the results in numpy format...")
        np.save(dataFilesRootpath + valuesCompleteName, dataV)
        np.save(dataFilesRootpath + labelsCompleteName, dataL)
        print("Numpy files saved correctly with names {} and {}".format(valuesCompleteName, labelsCompleteName))
    
    return dataV, dataL 


#    if(nTestUsers != 4):
#        fileExtensionName = '_' + str(nTestUsers) + 'testSubjects'

def divideDataSetForTrain(dataValues, dataLabels, nTestUsers=4):
    
    sisFallDatabaseInstance = FallDatabase()
    
    testPercentage = nTestUsers*100 / len(dataValues)
    print("Loading data for train, with {0} users for test ({1:.2f} % approximatelly)...".format(nTestUsers, testPercentage))
    
    sisFallDatabaseInstance = FallDatabase()
    
    trainData, trainClasif, testData, testClasif, dataOrganization = sisFallDatabaseInstance.structuredData(dataValues, dataLabels, nTestSubjects=nTestUsers)
    
    return trainData, trainClasif, testData, testClasif, dataOrganization

def prepareDataSetForTrain(dataFilesRootpath='./SisFallData/', nTestUsers=4):
    
    #  1st. Check if a divided dataset with same % of test users already exists
    #      A. Yes: Load and return it
    #      B. No. Then:
    #          B.1. Call to 'prepareDataset' function
    #          B.2. Call to 'divideDataSetForTrain' function
    #          B.3. Save it in file and return it
    
    dataForTrainFileName = 'SisFallDividedForTrain'
    fileExtensionName = ''
    fileFormat = '.npy'
    
    if(nTestUsers != 4):
        fileExtensionName = '_' + str(nTestUsers) + 'testSubjects'
        
    dataForTrainCompleteName = dataForTrainFileName + fileExtensionName + fileFormat
    
    if os.path.isfile(dataFilesRootpath + dataForTrainCompleteName):
        print("\nDivided (train - test) found. Loading...")
        
        dataDict = np.load(dataFilesRootpath + dataForTrainCompleteName)[()]
        trainData, trainClasif, testData, testClasif, dataOrganization = dataDict['trainData'],dataDict['trainClasif'],dataDict['testData'],dataDict['testClasif'],dataDict['dataOrganization']
        del dataDict
    else:
        print("\nDivided train - test data not found. Generating...")
        dataValues, dataLabels = prepareDataset(dataFilesRootpath)
        trainData, trainClasif, testData, testClasif, dataOrganization = divideDataSetForTrain(dataValues, dataLabels, nTestUsers)
        del dataValues
        del dataLabels
        np.save(dataFilesRootpath + dataForTrainCompleteName, {'trainData':trainData, 'trainClasif':trainClasif, 'testData':testData, 'testClasif':testClasif, 'dataOrganization':dataOrganization})
    

    return trainData, trainClasif, testData, testClasif, dataOrganization

def createDataSetInBlocks(trainData, trainClasif, testData, testClasif, dataOrganization, windowSize=128, stride=64, randomOrder = False):
    
    sisFallDatabaseInstance = FallDatabase()
    
    dataTrWinValues, dataTrWinLabel, dataTestWinValues, dataTestWinLabel, dataWinOrganization = sisFallDatabaseInstance.assignWindowLabel(trainData, trainClasif, testData, testClasif, dataOrganization, windowSize, stride, randomOrder = False)
    
    return dataTrWinValues, dataTrWinLabel, dataTestWinValues, dataTestWinLabel, dataWinOrganization

def loadDataSetInBlocks(dataFilesRootpath='./SisFallData/', nTestUsers=4, windowSize=128, stride=64, randomOrder = False):
    
    print("Loading data...")
    
    trainData, trainClasif, testData, testClasif, dataOrganization = prepareDataSetForTrain(dataFilesRootpath, nTestUsers)
    
    print("\nData loaded correctly\n")
        
    print("Estructuring data in blocks...\n")
    
    dataTrWinValues, dataTrWinLabel, dataTestWinValues, dataTestWinLabel, dataWinOrganization = createDataSetInBlocks(trainData, trainClasif, testData, testClasif, dataOrganization, windowSize, stride, randomOrder)

    print("Data generated correctly")
    
    return dataTrWinValues, dataTrWinLabel, dataTestWinValues, dataTestWinLabel, dataWinOrganization

def reduce_frequency_of_window_samples(win_values):
    win_values_with_lower_freq = list()
    for sample in win_values:
        sample_lower_freq = np.array([sample[i] for i in range(len(sample)) if i%2==0])
        win_values_with_lower_freq.append(sample_lower_freq)
    return np.array(win_values_with_lower_freq)

#########################################
    
# Weights:

#unique, counts = np.unique(dataTrWinLabel, return_counts=True)
#
#dict_counts = dict(zip(unique,counts))
#print(dict_counts)
#
#N_bkg = dict_counts[0]
#N_alert = dict_counts[1]
#N_fall = dict_counts[2]
#
#print(N_bkg, N_alert, N_fall)
#
#w_bkg = 1
#w_alert = N_bkg / N_alert 
#w_fall = N_bkg / N_fall
#
#target_weights = [w_bkg,w_alert,w_fall]
#print(target_weights)



###############################################################
#
# The loss function implemented doesn't work with int targets
#from keras.utils import to_categorical
#
#dataTrWinLabelOneHot = to_categorical(dataTrWinLabel)
#dataTestWinLabelOneHot = to_categorical(dataTestWinLabel)
#
###############################################################

