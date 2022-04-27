import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate, no_iterations):
        self.learning_rate = learning_rate
        self.no_iterations = no_iterations

    def sigmoid_function(self, formula):
        result = 1 / (1 + np.exp(-formula))
        return result

    def gradient_cost_function(self, weights, bias, x_val, y_val):
        m = x_val.shape[0]

        # Prediction
        final_result = self.sigmoid_function(np.dot(weights, x_val.T) + bias)
        # cost
        y_val_T = y_val.T
        cost = (-1 / m) * (np.sum((y_val_T * np.log(final_result)) + ((1 - y_val_T) * (np.log(1 - final_result)))))
        #

        # Gradient calculation
        dw = (1 / m) * (np.dot(x_val.T, (final_result - y_val.T).T))
        db = (1 / m) * (np.sum(final_result - y_val.T))

        grads = {"dw": dw, "db": db}

        return grads, cost

    def fit_function(self, X_value, Y_value):

        # Get number of features
        no_of_features = X_value.shape[1]
        weights = np.zeros((1, no_of_features))
        bias = 0
        costs = []

        for i in range(self.no_iterations):
            #
            grads, cost = self.gradient_cost_function(weights, bias, X_value, Y_value)
            #
            dw = grads["dw"]
            db = grads["db"]
            # weight update
            weights = weights - (self.learning_rate * (dw.T))
            bias = bias - (self.learning_rate * db)
            #

            if (i % 100 == 0):
                costs.append(cost)
                # print("Cost after %i iteration is %f" %(i, cost))

        # final parameters
        coeff = {"Weights": weights, "Bias": bias}
        gradient = {"dw": dw, "db": db}

        self.weights = coeff["Weights"]
        self.bias = coeff["Bias"]
        return coeff, gradient, costs

    def predict(self, pred_value):
        m = pred_value.shape[0]
        Y = np.dot(self.weights, pred_value.T) + self.bias
        pred_value = self.sigmoid_function(Y)

        y_pred = np.zeros((1, m))
        for i in range(pred_value.shape[1]):
            if pred_value[0][i] > 0.5:
                y_pred[0][i] = 1
            else:
                y_pred[0][i] = 0

        return y_pred

