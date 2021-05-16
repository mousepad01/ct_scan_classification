from dataIO import *
from sklearn import svm, preprocessing, metrics
import time
import random


'''trainImages, trainLabels = loadData(fileName="train.txt", samplesCnt=15000, samplesFolder="train", test=False)

validationImages, validationLabels = loadData(fileName="validation.txt", samplesCnt=4500,
                                                samplesFolder="validation", test=False)

testImages, imageNames = loadData(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

svc = svm.SVC(kernel='rbf')
svc.fit(trainImages, trainLabels)
predictedLabels = svc.predict(testImages)

predictions = []
for i in range(3900):
    predictions.append((imageNames[i], int(predictedLabels[i])))

putPredictions("svcRbfPredictions.txt", predictions)'''

'''for samplesCnt in range(100, 15000, 150):

    med = 0
    testCnt = 7

    for r in range(7):

        trainImages, trainLabels = loadData(fileName="train.txt", samplesCnt=15000, samplesFolder="train", test=False)

        validationImages, validationLabels = loadData(fileName="validation.txt", samplesCnt=4500,
                                                        samplesFolder="validation", test=False)

        bound = random.randint(0, 15000 - samplesCnt)
        trainImages = trainImages[bound:bound + samplesCnt]
        trainLabels = trainLabels[bound:bound + samplesCnt]

        t = time.time()

        svc = svm.SVC(kernel='rbf')
        svc.fit(trainImages, trainLabels)
        predictedLabels = svc.predict(validationImages)

        med += metrics.accuracy_score(validationLabels, predictedLabels)

        print(f"nr sample uri {samplesCnt}, acuratete: {metrics.accuracy_score(validationLabels, predictedLabels)}, timp: {time.time() - t}\n")

    print(f"========================\npentru nr sample uri {samplesCnt}, acuratetea medie: {med / testCnt}\n===========================\n")'''

log = open("logfile.txt", "w")

trainImagesFull, trainLabelsFull = loadData(fileName="train.txt", samplesCnt=15000, samplesFolder="train", test=False)

validationImages, validationLabels = loadData(fileName="validation.txt", samplesCnt=4500,
                                                  samplesFolder="validation", test=False)


def crange():

    c = [1, 3, 5, 6, 7, 10, 15, 20, 25, 30, 40, 60, 100, 150, 200, 300, 400, 1000,
         0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.001, 0.0001]

    g = ['scale', 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1]

    for val in c:
        for valg in g:
            yield valg, val


topParams = []
samplesCnt = 1500
reps = 1

for gamma, C in crange():

    acc = 0
    bs = []

    t = time.time()

    for _ in range(reps):

        bound = random.randint(0, 15000 - samplesCnt)
        trainImages = trainImagesFull[bound:bound + samplesCnt]
        trainLabels = trainLabelsFull[bound:bound + samplesCnt]

        bs.append(bound)

        svc = svm.SVC(kernel='rbf', C=C, gamma=gamma)

        svc.fit(trainImages, trainLabels)
        predictedLabels = svc.predict(validationImages)

        acc += metrics.accuracy_score(validationLabels, predictedLabels)

    acc /= reps

    topParams.append((acc, C, gamma, bs))

    log.write(f"acuratete C = {C}, gamma = {gamma}, bs = {bs}: {acc}, timp = {time.time() - t}\n")
    log.flush()


topParams.sort(key=lambda e: e[0], reverse=True)

log.write(f"topParams pt samplesCnt = {samplesCnt}: {topParams}\n")
log.flush()

log.write("\n\n====================================================================\n\n")
log.flush()

topParamsFull = []
for _, C, gamma, _ in topParams[:-4]:

    trainImages = trainImagesFull
    trainLabels = trainLabelsFull

    t = time.time()

    svc = svm.SVC(kernel='rbf', C=C)
    svc.fit(trainImages, trainLabels)
    predictedLabels = svc.predict(validationImages)

    acc = metrics.accuracy_score(validationLabels, predictedLabels)

    topParamsFull.append((acc, C, gamma))

    log.write(f"acuratete C = {C}, gamma = {gamma}: {acc}, timp = {time.time() - t}\n")
    log.flush()

topParamsFull.sort(key=lambda e: e[0], reverse=True)

log.write(f"topParams pt samplesCnt = 15000: {topParamsFull}\n\n")
log.flush()

log.write("finish")
log.close()





