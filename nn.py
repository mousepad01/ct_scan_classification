from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import tensorflow as tf
from dataIO import *


def first_alexnet_try():  # ~0.57 pe validation si test

    trainImagesFull, trainLabelsFull = loadDataMatrix4D(fileName="train.txt", samplesCnt=15000, samplesFolder="train",
                                                        test=False)

    validationImages, validationLabels = loadDataMatrix4D(fileName="validation.txt", samplesCnt=4500,
                                                          samplesFolder="validation", test=False)

    testImages, imageNames = loadDataMatrix4D(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

    trainImages = trainImagesFull
    trainLabels = trainLabelsFull

    tensorTrainImages = tf.convert_to_tensor(trainImages)
    tensorTrainLabels = tf.convert_to_tensor(trainLabels)
    tensorTrainLabels = tf.keras.utils.to_categorical(tensorTrainLabels, 3)

    tensorValidationImages = tf.convert_to_tensor(validationImages)
    tensorTestImages = tf.convert_to_tensor(testImages)

    tf.keras.backend.set_image_data_format("channels_first")

    # alexnet adaptat dupa https://analyticsindiamag.com/hands-on-guide-to-implementing-alexnet-with-keras-for-multi-class-image-classification/
    nn = Sequential(
        [

            Conv2D(filters=96, input_shape=(1, 50, 50), kernel_size=(11, 11), strides=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

            Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

            Flatten(),

            Dense(4096, input_shape=(1, 50, 50)),
            BatchNormalization(),
            Activation('relu'),

            Dropout(0.4),

            Dense(4096),
            BatchNormalization(),
            Activation('relu'),

            Dropout(0.4),

            Dense(1000),
            BatchNormalization(),
            Activation('relu'),

            Dropout(0.4),

            Dense(3),
            BatchNormalization(),
            Activation('softmax')
        ]
    )

    # nn.compile(optimizer='adam', metrics=['accuracy'], loss='mse')
    nn.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    nn.fit(tensorTrainImages, tensorTrainLabels, initial_epoch=0, epochs=100)

    validationPredictions = nn.predict_classes(tensorValidationImages)

    ok = 0
    for i in range(4500):
        if validationPredictions[i] == int(validationLabels[i]):
            ok += 1

    print(f"accuracy: {round(ok / 4500, 4)}")

    testPredictions = nn.predict_classes(tensorTestImages)

    predictions = []
    for i in range(3900):
        predictions.append((imageNames[i], testPredictions[i]))

    putPredictions("firsttry_alexnet_Predictions.txt", predictions)


def second_try_custom():  # 0.79 pe vaidation 0.76 pe test

    trainImagesFull, trainLabelsFull = loadDataMatrix4D(fileName="train.txt", samplesCnt=15000, samplesFolder="train",
                                                        test=False)

    validationImages, validationLabels = loadDataMatrix4D(fileName="validation.txt", samplesCnt=4500,
                                                          samplesFolder="validation", test=False)

    testImages, imageNames = loadDataMatrix4D(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

    trainImages = trainImagesFull
    trainLabels = trainLabelsFull

    tensorTrainImages = tf.convert_to_tensor(trainImages)
    tensorTrainLabels = tf.convert_to_tensor(trainLabels)
    tensorTrainLabels = tf.keras.utils.to_categorical(tensorTrainLabels, 3)

    tensorValidationImages = tf.convert_to_tensor(validationImages)
    tensorTestImages = tf.convert_to_tensor(testImages)

    tf.keras.backend.set_image_data_format("channels_first")

    nn = Sequential(
        [

            Conv2D(filters=64, input_shape=(1, 50, 50), kernel_size=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=128, kernel_size=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=384, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=128, kernel_size=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Flatten(),

            Dense(2048, input_shape=(1, 50, 50)),
            BatchNormalization(),
            Activation('relu'),

            Dense(1024),
            BatchNormalization(),
            Activation('relu'),

            Dense(256),
            BatchNormalization(),
            Activation('relu'),

            Dense(64),
            BatchNormalization(),
            Activation('relu'),

            Dense(16),
            BatchNormalization(),
            Activation('relu'),

            Dense(3),
            BatchNormalization(),
            Activation('softmax')
        ]
    )

    # nn.compile(optimizer='adam', metrics=['accuracy'], loss='mse')
    nn.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    nn.fit(tensorTrainImages, tensorTrainLabels, initial_epoch=0, epochs=70)

    validationPredictions = nn.predict_classes(tensorValidationImages)

    ok = 0
    for i in range(4500):
        if validationPredictions[i] == int(validationLabels[i]):
            ok += 1

    print(f"accuracy: {round(ok / 4500, 4)}")

    testPredictions = nn.predict_classes(tensorTestImages)

    predictions = []
    for i in range(3900):
        predictions.append((imageNames[i], testPredictions[i]))

    putPredictions("secondtry_custom_cnn_Predictions.txt", predictions)


def third_try_custom():
    trainImagesFull, trainLabelsFull = loadDataMatrix4D(fileName="train.txt", samplesCnt=15000, samplesFolder="train",
                                                        test=False)

    validationImages, validationLabels = loadDataMatrix4D(fileName="validation.txt", samplesCnt=4500,
                                                          samplesFolder="validation", test=False)

    testImages, imageNames = loadDataMatrix4D(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

    trainImages = trainImagesFull
    trainLabels = trainLabelsFull

    tensorTrainImages = tf.convert_to_tensor(trainImages)
    tensorTrainLabels = tf.convert_to_tensor(trainLabels)
    tensorTrainLabels = tf.keras.utils.to_categorical(tensorTrainLabels, 3)

    tensorValidationImages = tf.convert_to_tensor(validationImages)
    tensorTestImages = tf.convert_to_tensor(testImages)

    tf.keras.backend.set_image_data_format("channels_first")

    nn = Sequential(
        [

            Conv2D(filters=128, input_shape=(1, 50, 50), kernel_size=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=162, kernel_size=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=224, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=96, kernel_size=(4, 4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Flatten(),

            Dense(1024),
            BatchNormalization(),
            Activation('relu'),

            Dense(256),
            BatchNormalization(),
            Activation('relu'),

            Dense(64),
            BatchNormalization(),
            Activation('relu'),

            Dense(16),
            BatchNormalization(),
            Activation('relu'),

            Dense(3),
            BatchNormalization(),
            Activation('softmax')
        ]
    )

    # nn.compile(optimizer='adam', metrics=['accuracy'], loss='mse')
    nn.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    nn.fit(tensorTrainImages, tensorTrainLabels, initial_epoch=0, epochs=60)

    validationPredictions = nn.predict_classes(tensorValidationImages)

    ok = 0
    for i in range(4500):
        if validationPredictions[i] == int(validationLabels[i]):
            ok += 1

    print(f"accuracy: {round(ok / 4500, 4)}")

    testPredictions = nn.predict_classes(tensorTestImages)

    predictions = []
    for i in range(3900):
        predictions.append((imageNames[i], testPredictions[i]))

    putPredictions("thirdtry_custom_cnn_Predictions.txt", predictions)


def fourth_try_custom():

    trainImagesFull, trainLabelsFull = loadDataMatrix4D(fileName="train.txt", samplesCnt=15000, samplesFolder="train",
                                                        test=False)

    validationImages, validationLabels = loadDataMatrix4D(fileName="validation.txt", samplesCnt=4500,
                                                          samplesFolder="validation", test=False)

    testImages, imageNames = loadDataMatrix4D(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

    trainImages = trainImagesFull
    trainLabels = trainLabelsFull

    tensorTrainImages = tf.convert_to_tensor(trainImages)
    tensorTrainLabels = tf.convert_to_tensor(trainLabels)
    tensorTrainLabels = tf.keras.utils.to_categorical(tensorTrainLabels, 3)

    tensorValidationImages = tf.convert_to_tensor(validationImages)
    tensorTestImages = tf.convert_to_tensor(testImages)

    tf.keras.backend.set_image_data_format("channels_first")

    nn = Sequential(
        [

            Conv2D(filters=64, input_shape=(1, 50, 50), kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=100, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=200, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=64, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=100, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=200, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=100, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Flatten(),

            Dense(1024),
            BatchNormalization(),
            Activation('relu'),

            Dense(256),
            BatchNormalization(),
            Activation('relu'),

            Dense(64),
            BatchNormalization(),
            Activation('relu'),

            Dense(16),
            BatchNormalization(),
            Activation('relu'),

            Dense(3),
            BatchNormalization(),
            Activation('softmax')
        ]
    )

    nn.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    nn.fit(tensorTrainImages, tensorTrainLabels, initial_epoch=0, epochs=100)

    validationPredictions = nn.predict_classes(tensorValidationImages)

    ok = 0
    for i in range(4500):
        if validationPredictions[i] == int(validationLabels[i]):
            ok += 1

    print(f"accuracy: {round(ok / 4500, 4)}")

    testPredictions = nn.predict_classes(tensorTestImages)

    predictions = []
    for i in range(3900):
        predictions.append((imageNames[i], testPredictions[i]))

    putPredictions("fourthtry_custom_cnn_Predictions.txt", predictions)


def fifth_try_custom():


    trainImagesFull, trainLabelsFull = loadDataMatrix4D(fileName="train.txt", samplesCnt=15000, samplesFolder="train",
                                                        test=False)

    validationImages, validationLabels = loadDataMatrix4D(fileName="validation.txt", samplesCnt=4500,
                                                          samplesFolder="validation", test=False)

    testImages, imageNames = loadDataMatrix4D(fileName="test.txt", samplesCnt=3900, samplesFolder="test", test=True)

    trainImages = trainImagesFull
    trainLabels = trainLabelsFull

    tensorTrainImages = tf.convert_to_tensor(trainImages)
    tensorTrainLabels = tf.convert_to_tensor(trainLabels)
    tensorTrainLabels = tf.keras.utils.to_categorical(tensorTrainLabels, 3)

    tensorValidationImages = tf.convert_to_tensor(validationImages)
    tensorTestImages = tf.convert_to_tensor(testImages)

    tf.keras.backend.set_image_data_format("channels_first")

    nn = Sequential(
        [

            Conv2D(filters=32, input_shape=(1, 50, 50), kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=48, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=64, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=128, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(filters=394, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=128, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(filters=64, kernel_size=(2, 2)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Flatten(),

            Dense(1024),
            BatchNormalization(),
            Activation('relu'),

            Dense(256),
            BatchNormalization(),
            Activation('relu'),

            Dense(64),
            BatchNormalization(),
            Activation('relu'),

            Dense(16),
            BatchNormalization(),
            Activation('relu'),

            Dense(3),
            BatchNormalization(),
            Activation('softmax')
        ]
    )

    nn.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    nn.fit(tensorTrainImages, tensorTrainLabels, initial_epoch=0, epochs=10)

    validationPredictions = nn.predict_classes(tensorValidationImages)

    ok = 0
    for i in range(4500):
        if validationPredictions[i] == int(validationLabels[i]):
            ok += 1

    print(f"accuracy: {round(ok / 4500, 4)}")

    testPredictions = nn.predict_classes(tensorTestImages)

    predictions = []
    for i in range(3900):
        predictions.append((imageNames[i], testPredictions[i]))

    putPredictions("fifthtry_custom_cnn_Predictions.txt", predictions)

#first_alexnet_try()
#second_try_custom()
#third_try_custom()
fourth_try_custom()
#fifth_try_custom()




