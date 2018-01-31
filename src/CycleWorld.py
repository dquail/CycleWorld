#!/usr/bin/env python

"""
Author: David Quail, September, 2017.
Description:
Creates the reinformement learning environment for taking actions, receiving state and rewards.

"""

import numpy

from CycleState import *

NUMBER_OF_STATES = 4
SPECIAL_STATE_INDEX = 0
class CycleWorld:


    def __init__(self):
        self.isCreated = True
        self.states = []

        #Create Grids for each
        self.currentIndex = 0

        for i in range(NUMBER_OF_STATES):
            s = CycleState()
            if i == SPECIAL_STATE_INDEX:
                s.isSpecial = True
            self.states.append(s)

        self.states[3].color = 'g'

    def printWorld(self):

        printString = ""
        for i in range(len(self.states)):
            str = ""
            state = self.states[i]
            color = state.color

            if state.isSpecial:
                print("I'm special!!!!")
                color = color.capitalize()

            str = color
            #Check if agent is there
            if i == self.currentIndex:
                str = str + ">"


            printString = printString + str

        print(printString)

    """
    Creates a bit vector of length 9 with the following bit values:
    - bit 0: 1 if color is r
    - bit 1: 1 if color is g
    - bit 2: 1 if color is b
    - bit 3: 1 if color is w
    - bit 4: 1 (bias bit)
    - bit 5-8: randomly 0 or 1
    """
    def createObservation(self, oldState, newState):
        #Currently doesn't use any info from the state transitioned from

        vector = numpy.zeros(5)

        #Set color bits
        color = newState.color
        if color == "r":
            vector[0] = 1
        elif color == "g":
            vector[1] = 1
        elif color == "b":
            vector[2] = 1
        elif color == "w":
            vector[3] = 1

        #set bias bit
        vector[4] = 1


        randomVector = numpy.random.randint(2, size = 4)

        return numpy.append(vector, randomVector)


    """
    Returns a tuple containing observation (vector), reward (scalar))
    Actions are L (turn left), R (turn right)
    """
    def takeAction(self, action):


        currentState=  self.states[self.currentIndex]
        nextState = ""

        #Determine reward. 1 unless you are in the special state and choose left

        reward = 0
        if action == 'R':
            reward = 0
        if action == 'L':
            reward = -10
            if currentState.isSpecial:
                reward = 10

        #Get observation
        if self.currentIndex == len(self.states) - 1:
            #At the end of the array. Move to the beginning
            nextState = self.states[0]
            self.currentIndex = 0
        else:
            #not at teh end of array. Move to next
            nextState = self.states[self.currentIndex + 1]
            self.currentIndex = self.currentIndex + 1

        observation = self.createObservation(currentState, nextState)
        self.printWorld()
        return (reward, observation)


