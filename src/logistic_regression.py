from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()

def print_shape():
    print(digits.data.shape)

def plot_digits():
    pyplot.figure(figsize=(20,4))
    for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
        pyplot.subplot(1, 5, index + 1)
        pyplot.imshow(np.reshape(image, (8,8)), cmap=pyplot.cm.gray)
        pyplot.title('Training: %i\n' % label, fontsize = 20)
    
    pyplot.show()

def print_misclassified_image(actual_y, predicted_y):
    index = 0
    misclassifiedImages = []
    for label, predict in zip(actual_y, predicted_y):
        if label != predict: 
            misclassifiedImages.append(index)
            index +=1

    print("Misclassified images: ", misclassifiedImages)

plot_digits()

# Training
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# Predictions
logisticRegressor = LogisticRegression(multi_class="auto", solver="lbfgs")

logisticRegressor.fit(x_train, y_train)

prediction_1 = logisticRegressor.predict(x_test[0].reshape(1, -1))

prediction_2 = logisticRegressor.predict(x_test[0:10])

prediction_3 = logisticRegressor.predict(x_test)

print(prediction_2)

# Score
score = logisticRegressor.score(x_test, y_test)

print(score)

# Just for interest show misclassified images
print_misclassified_image(y_test, prediction_3)
