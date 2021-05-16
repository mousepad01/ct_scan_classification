from dataIO import *
from sklearn.neighbors import KNeighborsClassifier


def l1(x, y):
    dArr = np.abs(x - y)
    return np.sum(dArr)


def l2(x, y):
    dArr = np.square(x - y)
    return np.sqrt(np.sum(dArr))


class KnnClassifier:

    def __init__(self, trainImages, trainLabels):

        self.trainImages = trainImages
        self.trainLabels = trainLabels

        self.trainCnt = self.trainImages.shape[0]

    distance = {'l2': l2, 'l1': l1}

    def classifyImage(self, testImage, numNeighbours, metric):

        d = KnnClassifier.distance[metric]

        mins = [[float('inf'), None] for _ in range(numNeighbours)]

        dst = np.empty(self.trainCnt)
        for i in range(len(dst)):
            dst[i] = d(self.trainImages[i], testImage)

        for i in range(len(dst)):

            j = 0
            while j < numNeighbours and mins[j][0] < dst[i]:
                j += 1

            if j < numNeighbours and dst[i] <= mins[j][0]:

                for k in range(numNeighbours - 1, j, -1):
                    mins[k][0] = mins[k - 1][0]
                    mins[k][1] = mins[k - 1][1]

                mins[j][0] = dst[i]
                mins[j][1] = i

        cnts = [0 for _ in range(3)]
        for nb in mins:
            cnts[int(self.trainLabels[nb[1]])] += 1

        m = -1
        maxLabel = None
        for i in range(3):

            if m < cnts[i]:
                m = cnts[i]
                maxLabel = i

        return maxLabel

    def classifyValidationImages(self, validationImages, validationLabels, numNeighbours, metric):

        predictedLabels = np.empty(validationImages.shape[0])

        for i in range(len(predictedLabels)):
            predictedLabels[i] = self.classifyImage(validationImages[i], numNeighbours=numNeighbours, metric=metric)

        okCnt = 0
        for i in range(len(predictedLabels)):
            if predictedLabels[i] == validationLabels[i]:
                okCnt += 1

        accuracy = okCnt / len(predictedLabels)
        print(f"acuratetea pentru numNeighbours = {numNeighbours} si metric = {metric} este de {accuracy}\n")

        return numNeighbours, accuracy



trainImages, trainLabels = loadData(fileName="train.txt", samplesCnt=15000, samplesFolder="train", test=False)

validationImages, validationLabels = loadData(fileName="validation.txt", samplesCnt=4500,
                                                samplesFolder="validation", test=False)

testImages, imageNames = loadData(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

# knnClassifier = KnnClassifier(trainImages, trainLabels)
# knnClassifier.classifyValidationImages(validationImages, validationLabels, numNeighbours=5, metric='l2')
# aceeasi precizie ca cel din biblioteca dar mai lent


'''for nn in {1, 3, 5, 7}:

    classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    classifier.fit(trainImages, trainLabels)

    print(classifier.score(validationImages, validationLabels))'''

classifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
classifier.fit(trainImages, trainLabels)
predictedLabels = classifier.predict(testImages)

predictions = []
for i in range(3900):
    predictions.append((imageNames[i], int(predictedLabels[i])))

putPredictions("knnPredictions.txt", predictions)
