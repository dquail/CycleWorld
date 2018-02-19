from pylab import *
import numpy


"""
BUG TODO - 
"for indicie" loops seems to assume that the state vector is a list of indicies that are "on"
That's not currently the format of the states coming into the function. So need to update this. 
"""

class DoubleQ:
    def __init__(self, alpha, eps, numberOfFeatures, numberOfActions):
        self.numberOfFeatures = numberOfFeatures
        self.actionOffset = numberOfFeatures
        self.numberOfWeights = numberOfFeatures * numberOfActions
        self.alpha = alpha
        self.eps = eps

        #TODO - Change after testing
        self.theta1 = numpy.zeros(self.numberOfWeights)
        self.theta2 = numpy.zeros(self.numberOfWeights)
        """
        self.theta1 = -0.001 * rand(self.numberOfWeights)
        self.theta2 = -0.001 * rand(self.numberOfWeights)
        """

    def resetQ(self):
        #TODO - Change after testing
        self.theta1 = numpy.zeros(self.numberOfWeights)
        self.theta2 = numpy.zeros(self.numberOfWeights)
        """
        self.theta1 = -0.001 * rand(self.numberOfWeights)
        self.theta2 = -0.001 * rand(self.numberOfWeights)
        """

    def qHat(self, state, action, theta):
        value = 0
        """
        releventTheta = theta[action * self.actionOffset : action * self.actionOffset + self.numberOfFeatures]
        value = np.inner(state, releventTheta)
        """
        indexes =  [i for i, x in enumerate(state) if x == 1.0]
        for indicie in indexes:
            value += theta[indicie + action * self.actionOffset]

        return value

    def bestAction(self, state, theta):
        # BUG below - this is hard coded with two actions. Should depend on number of a ctions
        return numpy.argmax([self.qHat(state, 0, theta), self.qHat(state, 1, theta)])

    def policy(self, state):
        # Combine the average of the two weight vectors
        theta = np.add(self.theta1, self.theta2) / 2
        #BUG below - this is hard coded with two actions. Should depend on number of a ctions
        return numpy.argmax([self.qHat(state, 0, theta), self.qHat(state, 1, theta)])

    def learn(self, state, action, nextState, reward):
        # determine which theta to update
        newStateValue = 0
        # tileIndices = state

        if (random() > 0.5):
            # print("Using theta1")
            nextStateValue = 0
            if not nextState is None:
                # Non terminal
                theta1BestNextAction = self.bestAction(nextState, self.theta1)
                newStateValue = self.qHat(nextState, theta1BestNextAction, self.theta2)

            learningError = self.alpha * (reward + newStateValue - self.qHat(state, action, self.theta1))
            indexes = [i for i, x in enumerate(state) if x == 1.0]
            for indicie in indexes:
                self.theta1[indicie + action * self.actionOffset] += learningError
            """
            #self.theta1[action * self.actionOffset: action * self.actionOffset + self.numberOfFeatures] += learningError
            """
        else:
            # print("Using theta2")
            nextStateValue = 0
            if not nextState is None:
                # Non terminal
                theta2BestNextAction = self.bestAction(nextState, self.theta2)
                newStateValue = self.qHat(nextState, theta2BestNextAction, self.theta1)

            learningError = self.alpha * (reward + newStateValue - self.qHat(state, action, self.theta2))
            indexes = [i for i, x in enumerate(state) if x == 1.0]
            for indicie in indexes:
                self.theta2[indicie + action * self.actionOffset] += learningError

            #self.theta2[action * self.actionOffset: action * self.actionOffset + self.numberOfFeatures] += learningError