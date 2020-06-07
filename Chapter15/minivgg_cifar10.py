import matplotlib
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from minivgg import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tensorflow.keras.models import load_model

matplotlib.use("Agg")
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# Load cifar10 from keras datasets
print("[INFO] loading CIFAR-10 dataset ...")
((trainData, trainLabels), (testData, testLabels)) = cifar10.load_data()

# scale input image to range [0, 1] for normalization
trainData = trainData.astype("float") / 255.0
testData = testData.astype("float") / 255.0

lb = LabelBinarizer()
trainLabels = lb.fit_transform(trainLabels)
testLabels = lb.transform(testLabels)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Build our model with SGD optimizer using decay, momentum and nesterov tuning parameters
# Decay is a paramemter to slowly reduce the initial learning rate over time to reduce overfitting - the smaller
# the learning rate -> the smaller the weight updates. Common setting for decay is to divide the initial learning
# rate with the number of epochs
# We build the model with the shape of our input images (32x32x3) and 10 clases from cifar10
print("[INFO] compiling model ...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network .. ")
H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), batch_size=64, epochs=40, verbose=1)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

print("[INFO] evaluating network ..")
predictions = model.predict(testData, batch_size=64)
print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

