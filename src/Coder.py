"""
Author: David Quail, September, 2017.
Description:
Creates the feature vector for reinforcement learning in the compass world.

"""

import numpy

class Coder:
    @staticmethod
    def getIndexes():
        return [1,2,3]

    @staticmethod
    def getVector():
        return [0, 0, 1, 0]

    """
    Returns a bit vector with following format:
    [isRed, isOrange, isYellow, isGreen, isBlue, isWhite, isOther]
    """
    @staticmethod
    def encodeColor(color):
        v = numpy.zeros(6)
        if (color == "red"):
            v[0] = 1
        elif (color == "orange"):
            v[1] = 1
        elif (color == "yellow"):
            v[2] = 1
        elif (color == "green"):
            v[3] = 1
        elif (color == "blue"):
            v[4] = 1
        elif (color == "white"):
            v[5] = 1
        else:
            v[6] = 1
        return v

    """
    Returns a vector with the following format:
    [0 steps, 1 steps, 2 steps, 3 steps, 4 steps, 5 steps, more steps]
    """
    @staticmethod
    def encodeSteps(steps):
        v = numpy.zeros(6)
        index = 0

        if steps > 5:
            index = 5
        else:
            index = steps

        v[index] = 1

        return v




