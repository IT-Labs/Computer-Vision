from nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print("[INFO] loading MNIST dataset..")
digits = datasets.load_digits()
data = digits.data.astype("float")
# apply normalization over the data using min/max scaling to range of [0,1]
data = (data - data.min()) / (data.max() - data.min())

print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

# split test/train
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# transforms the shape of the array of labels that are benefical for our network
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network..")
# define the architecture of the neural network
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY)

print("[INFO] evaluation network")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))