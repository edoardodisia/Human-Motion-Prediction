import os
import time
import json
import numpy as np
import math


class FileManager:
    OUTPUT_FOLDER = os.getcwd() + "/Output_files"
    POSE3D_DATA_FOLDER = "/pose3d_data"
    MULTI_SEGMENTS_LABELS = ["segment", "frame", "joint", "x", "y", "z"]
    SINGLE_SEGMENT_LABLES = ["frame", "joint", "x", "y", "z"]

    def SaveToTxt(poseData, fileName, jointsOfInterest = []):
        # check if root path exists
        if (not os.path.isdir(FileManager.OUTPUT_FOLDER)):
            os.mkdir(FileManager.OUTPUT_FOLDER)

        # check if path exits
        path = FileManager.OUTPUT_FOLDER + FileManager.POSE3D_DATA_FOLDER
        if (not os.path.isdir(path)):
            os.mkdir(path)

        # check if the filename contains extension
        if ".txt" in fileName:
            name = "/" + fileName + " " + time.strftime("%Y%m%d_%H%M%S")
        else:
            name = "/" + fileName + " " + time.strftime("%Y%m%d_%H%M%S") + ".txt"  

        # open file in writing mode
        file = open(path + name, 'w')

        # check data input dimensions
        if (len(poseData.shape) == 3):
            if (poseData.shape[1] < poseData.shape[2]):
                tempData = poseData.transpose(1, 2)[0]
            else:
                tempData = poseData[0]
        elif (len(poseData.shape) == 2):
            tempData = poseData

        # creation of the lines list of the file body to save
        lines = []
        for frameNbr in range(tempData.shape[1]):
            lines.append('Frame ' + str(frameNbr + 1) + "\n")

            # interface variables to manage the joints of interest list
            ptrJointOfInt = 0
            ptrPoseJoint = 0
            jointsPose = range(0, int(tempData.shape[0]/3))

            for ptrQuotes in range(0, tempData.shape[0], 3):
                # check if the joint is in the list of interest (i.e to be saved or not)
                if ((len(jointsOfInterest) > 0 and 
                     ptrJointOfInt < len(jointsOfInterest) and
                     ptrPoseJoint < len(jointsPose) and
                     jointsOfInterest[ptrJointOfInt] == jointsPose[ptrPoseJoint]) 
                    or (len(jointsOfInterest) == 0)):
                    x = tempData[ptrQuotes, frameNbr]
                    y = tempData[ptrQuotes + 1, frameNbr]
                    z = tempData[ptrQuotes + 2, frameNbr]

                    quoteX = "{:10.3f}".format(x) 
                    quoteY = "{:10.3f}".format(y)
                    quoteZ = "{:10.3f}".format(z)  

                    if (ptrQuotes == 0):
                        jointRef = "J" + str(ptrQuotes)
                    else:
                        jointRef = "J" + str(int(ptrQuotes / 3))
                    lines.append("\t" + jointRef + "\t" + str([quoteX, quoteY, quoteZ]) + "\n") 
                    ptrJointOfInt = ptrJointOfInt + 1

                ptrPoseJoint = ptrPoseJoint + 1    

        file.writelines(lines)

    def LoadFromTxt(fileName):
        raise NotImplementedError

    def SerializeJson(data, fileName, labels : list = [], additionalPath : str = ""):
        # check if root path exists
        if (not os.path.isdir(FileManager.OUTPUT_FOLDER)):
            os.mkdir(FileManager.OUTPUT_FOLDER)
        
        # check if full path exists
        if (additionalPath == ""):
            path = FileManager.OUTPUT_FOLDER + FileManager.POSE3D_DATA_FOLDER
        else:
            if ("/" in additionalPath):
                path = FileManager.OUTPUT_FOLDER + additionalPath
            else:
                path = FileManager.OUTPUT_FOLDER + "/" + additionalPath

        if (not os.path.isdir(path)):
            os.mkdir(path)

        # check if the filename contains extension
        if ".json" in fileName:
            name = fileName + " " + time.strftime("%Y%m%d_%H%M%S")
        else:
            name = fileName + " " + time.strftime("%Y%m%d_%H%M%S") + ".json"

        # open file in writing mode
        file = open(path + "/" + name, "w")

        # verifica tipo di dato
        dictData = {}

        if (not isinstance(data, dict)):
            FileManager.FromMatrixToDict(data, dictData, labels)  
        else: 
            dictData = data

        json.dump(dictData, file, indent = 4)

    def FromMatrixToDict(matrix, dictData, labels):   
        # scorrimento prima dimensione tensore
        for i in range(matrix.shape[0]):
            # se la dimensione è maggiore di 1 significa è necessario scendere di un livello
            # tramite ricorsione per associare correttamente chiave-valore
            if (len(matrix.shape) > 1):
                dictData[labels[0] + str(i)] = dict()
                FileManager.FromMatrixToDict(matrix[i], dictData[labels[0] + str(i)], labels[1:len(labels)])  
            else:
                # associazione chiave-valore     
                dictData[labels[i]] = str(matrix[i])

    def DeserializeJson(fileName, subFolder = "", dataAsDict = False):
        # verifica se è stata specificata l'estensione
        if ".json" in fileName:
            name = fileName
        else:
            name = fileName + ".json"

        if (subFolder == ""):
            path = FileManager.OUTPUT_FOLDER + FileManager.POSE3D_DATA_FOLDER + "/" + name
        else:
            if ("/" in subFolder):
                path = FileManager.OUTPUT_FOLDER + subFolder + "/" + name
            else:
                path = FileManager.OUTPUT_FOLDER + "/" + subFolder + "/" + name

        if (os.path.isfile(path)):
            file = open(path, "r")
            dictData = json.load(file)

            # returning directly "raw" data
            if (dataAsDict):
                return dictData

            #calcolo dimensioni necessarie per rappresentare la matrice salvata nel dizionario
            dictDims = []
            FileManager.GetDictDimensions(dictData, dictDims)

            # creazione matrice con dimensioni richieste
            matrix = np.array(np.ndarray(dictDims))

            # calcolo totale valori da salvare nella matrice
            values = []
            FileManager.FromDictToList(dictData, values)

            # Hp: matrix => penultima dimensione = frame catturati
            #            => ultima dimensione = coordinate joint singolo frame
            totJoints = matrix.shape[len(matrix.shape) - 2]

            if (len(matrix.shape) == 4):
                startIndex = 0
                endIndex = totJoints * 3

                for segment in range(matrix.shape[0]):
                    for frame in range(matrix.shape[1]):
                        matrix[segment, frame, :, :] = np.array(values[startIndex : endIndex]).reshape(totJoints, 3)
                        startIndex = startIndex + totJoints * 3
                        endIndex = endIndex + totJoints * 3
            elif (len(matrix.shape) == 3):
                startIndex = 0
                endIndex = totJoints * 3

                for frame in range(matrix.shape[0]):
                    matrix[frame, :, :] = np.array(values[startIndex : endIndex]).reshape(totJoints, 3)
                    startIndex = startIndex + totJoints * 3
                    endIndex = endIndex + totJoints * 3
            else:
                raise NotImplementedError

        return matrix
        
    def GetDictDimensions(dictData : dict, listDims : list):
        # aggiunta dimensione dizionario alla lista
        listDims.append(len(dictData))

        # estrazione prima chiave dizionario
        for key in dictData:
            break

        # verifica se è necessaria ricorsione (dizionario di dizionario...)
        if (isinstance(dictData[key], dict)):
            FileManager.GetDictDimensions(dictData[key], listDims)

    def FromDictToList(dictData : dict, listValues : list):
        # scorrimento chiavi nel dizionario per salvataggio nella matrice
        for key in dictData:
            # verifica se è necessaria ricorsione
            if (isinstance(dictData[key], dict)):   
                FileManager.FromDictToList(dictData[key], listValues)
            else:
                listValues.append(float(dictData[key]))

class DataAnalyzer:
    def CalculateStatistics(samples : list):
        sum = 0
        for sample in samples:
            sum = sum + sample
        
        # mean value
        mu = sum / len(samples)

        sum = 0
        for sample in samples:
            sum = sum + pow((mu - sample), 2)

        # variance
        sigma2 = sum / len(samples)
        
        # standard deviation
        sigma = math.sqrt(sigma2)

        return mu, sigma2, sigma

    