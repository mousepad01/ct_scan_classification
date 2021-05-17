import numpy as np
from matplotlib import image
import os
from shutil import copyfile


def loadData(fileName, samplesCnt, samplesFolder, test):

    PIXEL_CNT = 2500

    file = open(fileName, "r")
    data = file.read()

    data = data.split('\n')
    data = data[:-1]

    currentPath = os.path.abspath(os.getcwd())

    if not test:

        for i in range(len(data)):
            data[i] = data[i].split(',')

        images = np.empty((samplesCnt, PIXEL_CNT))
        labels = np.empty(samplesCnt)

        for i in range(samplesCnt):

            imgname, label = data[i]
            img = image.imread(f"{currentPath}\\{samplesFolder}\\{imgname}")

            images[i] = np.reshape(img, PIXEL_CNT)
            labels[i] = label

        return images, labels

    else:

        images = np.empty((samplesCnt, PIXEL_CNT))

        for i in range(samplesCnt):

            img = image.imread(f"{currentPath}\\{samplesFolder}\\{data[i]}")
            images[i] = np.reshape(img, PIXEL_CNT)

        return images, data


def loadDataMatrix4D(fileName, samplesCnt, samplesFolder, test):

    INPUT_SHAPE = (samplesCnt, 1, 50, 50)

    file = open(fileName, "r")
    data = file.read()

    data = data.split('\n')
    data = data[:-1]

    currentPath = os.path.abspath(os.getcwd())

    if not test:

        for i in range(len(data)):
            data[i] = data[i].split(',')

        images = np.empty(INPUT_SHAPE)
        labels = np.empty(samplesCnt)

        for i in range(samplesCnt):
            imgname, label = data[i]
            img = image.imread(f"{currentPath}\\{samplesFolder}\\{imgname}")
            # img = np.pad(img, pad_width=((87, 87), (87, 87)), mode='constant')

            images[i] = img
            labels[i] = label

        return images, labels

    else:

        images = np.empty(INPUT_SHAPE)

        for i in range(samplesCnt):
            img = image.imread(f"{currentPath}\\{samplesFolder}\\{data[i]}")
            images[i][0] = img

        return images, data


def putPredictions(fileName, predictions):

    output = open(fileName, "w")
    output.write("id,label\n")

    for e in predictions:
        output.write(f"{e[0]},{e[1]}\n")

    output.close()


def splitData(fileName="train.txt", samplesCnt=15000):

    file = open(fileName, "r")
    data = file.read()

    data = data.split('\n')
    data = data[:-1]

    for i in range(len(data)):
        data[i] = data[i].split(',')

    currentPath = os.path.abspath(os.getcwd())

    os.mkdir(f"{currentPath}\\train{i}")

    for i in range(samplesCnt):

        imgname, label = data[i]
        copyfile(f"{currentPath}\\train\\{imgname}", f"{currentPath}\\train{int(label)}\\{imgname}")
