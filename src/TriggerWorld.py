#!/usr/bin/env python

"""
Author: David Quail, February, 2018.
Description:
Creates the reinformement learning environment for taking actions, receiving state and rewards.
Circle world where each state can take two actions:
    - Forward - Which moves to the next state in the cycle
    - Trigger - Which will either transition back to its own state or else to the terminal state.
"""

import numpy

from TriggerState import *

NUMBER_OF_STATES = 4
TERMINAL_STATE_INDEX = 30 # temporary hack so that there is no terminal state

class TriggerWorld:


    def __init__(self):
        self.isCreated = True
        self.states = []

        #Create Grids for each
        self.currentIndex = 0

        for i in range(NUMBER_OF_STATES):
            s = TriggerState()
            if i == TERMINAL_STATE_INDEX:
                s.triggerIsTerminal = True
            self.states.append(s)

        self.states[3].color = 'g'

    def printWorld(self):

        printString = ""
        for i in range(len(self.states)):
            str = ""
            state = self.states[i]
            color = state.color

            if state.triggerIsTerminal:
                color = color.capitalize()

            str = color
            #Check if agent is there
            if i == self.currentIndex:
                str = str + ">"


            printString = printString + str

        print(printString)

    def createObservation(self, newState):
        if newState == None:
            return None
        else:
            #return self.createObservationWithBiasAndRandom(oldState, newState)
            return  self.createObservationWithoutBiasAndRandom(newState)

    def createObservationWithoutBiasAndRandom(self, newState):
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

        #set bias bit to 0
        vector[4] = 0


        #randomVector = numpy.random.randint(2, size = 4)
        randomVector = numpy.zeros(4)

        observation = numpy.append(vector, randomVector)
        return observation

    """
    Creates a bit vector of length 9 with the following bit values:
    - bit 0: 1 if color is r
    - bit 1: 1 if color is g
    - bit 2: 1 if color is b
    - bit 3: 1 if color is w
    - bit 4: 1 (bias bit)
    - bit 5-8: randomly 0 or 1
    """
    def createObservationWithBiasAndRandom(self, oldState, newState):
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
    Actions are T (pull trigger), M (move))
    """
    def takeAction(self, action):


        currentState=  self.states[self.currentIndex]
        nextState = ""

        #Determine reward. 1 unless you are in the special state and choose left

        reward = -1

        if (action == 'M'):
            #Get observation
            if self.currentIndex == len(self.states) - 1:
                #At the end of the array. Move to the beginning
                nextState = self.states[0]
                self.currentIndex = 0
            else:
                #not at teh end of array. Move to next
                nextState = self.states[self.currentIndex + 1]
                self.currentIndex = self.currentIndex + 1

        elif (action == 'T'):
            if (self.currentIndex == TERMINAL_STATE_INDEX):
                nextState = None
            else:
                nextState = currentState

        observation = self.createObservation(nextState)

        return (reward, observation)


