import numpy as np
import pandas as pd

MAX_ITER = 20
NUMBER_OF_CLASSES = 3


class Perceptron:
    weights = np.zeros(4)
    bias = 0

    def __init__(self, reg=0):
        self.reg_coefficient = reg

    def train(self, data, labels):

        for _ in range(MAX_ITER):
            for i in range(len(data)):

                row = data[i]
                label = labels[i]

                weighted_sum = np.dot(self.weights.T, row) + self.bias

                if label * weighted_sum <= 0:
                    self.weights = ((1 - 2 * self.reg_coefficient) * self.weights) + (label * row)
                    self.bias = (1 - 2 * self.reg_coefficient) * self.bias + label

    def _predict(self, data):
        if np.dot(self.weights.T, data) + self.bias > 0:
            return 1
        else:
            return -1

    def predict(self, data):
        preds = [self._predict(x) for x in data]
        return preds

    def predict_confidence(self, data):
        return np.dot(self.weights.T, data) + self.bias


def split_data(data):
    x_values = data[:, 0:4]
    y_values = data[:, 4]

    return x_values, y_values


def change_to_binary(data, class1, class2=None):
    if class2 is None:
        two_class_data = [[x[0], x[1], x[2], x[3], 1] if x[4] == class1 else [x[0], x[1], x[2], x[3], -1]
                          for x in data]

    else:
        two_class_data = [[x[0], x[1], x[2], x[3], 1] if x[4] == class1 else [x[0], x[1], x[2], x[3], -1]
                          for x in data if (x[4] == class1 or x[4] == class2)]

    return np.array(two_class_data)


def build_multiclass(train, n=3, r=0):
    one_v_all = []
    for c1 in range(1, n+1):
        # Split data to binary

        train_x, train_y = split_data(change_to_binary(train, c1))

        p = Perceptron(reg=r)

        p.train(train_x, train_y)

        one_v_all.append(p)

    return one_v_all


def test_multiclass(test, perceptron_array):
    test_x, test_y = split_data(test)

    correct = 0

    for i in range(len(test_x)):

        features = test_x[i]
        actual = test_y[i]

        confidence = []

        for p in perceptron_array:
            confidence.append(p.predict_confidence(features))

        argmax = np.array(confidence).argmax() + 1

        if argmax == int(actual):
            correct += 1

    accuracy = round(correct / len(test_y), ndigits=4) * 100

    return accuracy


def accuracy(pred, actual):
    correct = sum([1 if x == y else 0 for (x, y) in zip(actual, pred)])
    accuracy = (correct / len(pred)) * 100

    accuracy = round(accuracy, ndigits=3)

    return accuracy


def main():
    train_data = pd.read_csv("train.data", header=None).to_numpy()
    test_data = pd.read_csv("test.data", header=None).to_numpy()

    out_string = "OUTPUT: \n\n BINARY_OUTPUT: \n"

    # rename class-x to x
    train_data = np.array([[x[0], x[1], x[2], x[3], int(x[4][6])] for x in train_data])
    test_data = np.array([[x[0], x[1], x[2], x[3], int(x[4][6])] for x in test_data])

    # Create all 1 v 1 perceptron
    output_binary = pd.DataFrame(columns=['Model', 'Train Accuracy', 'Test Accuracy'])

    for c1 in range(1, NUMBER_OF_CLASSES + 1):
        for c2 in range(c1, NUMBER_OF_CLASSES + 1):

            # If class 1 equals class 2 ignore this case
            if c1 == c2:
                continue

            model_name = "Class " + str(c1) + " vs Class " + str(c2)

            # Split data to binary

            train_x, train_y = split_data(change_to_binary(train_data, c1, class2=c2))
            test_x, test_y = split_data(change_to_binary(test_data, c1, class2=c2))

            p = Perceptron()

            p.train(train_x, train_y)

            train_preds = p.predict(train_x)
            test_preds = p.predict(test_x)

            train_acc = accuracy(train_preds, train_y)
            test_acc = accuracy(test_preds, test_y)

            # Update output dataframe

            output_binary.loc[len(output_binary.index)] = [model_name, train_acc, test_acc]

    print("Binary:\n", output_binary, "\n")

    # Build 1 v all classifiers with

    output_multi = pd.DataFrame(columns=['Regularisation Coefficient', 'Train Accuracy', 'Test Accuracy'])

    reg_coefficients = [0, 0.01, 0.1, 1, 10, 100]

    for reg in reg_coefficients:
        per = build_multiclass(train_data, r=reg)

        train_acc = test_multiclass(train_data, per)

        test_acc = test_multiclass(test_data, per)

        output_multi.loc[len(output_multi.index)] = [reg, train_acc, test_acc]

    print("Multi-class:\n", output_multi)

    out_string += output_binary.to_string(header=True) + " \n\n MULTI-CLASS \n" + output_multi.to_string(header=True)

    with open("data.out", 'a') as f:
        f.write(out_string)


if __name__ == "__main__":
    main()
