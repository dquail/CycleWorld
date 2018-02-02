#!/usr/bin/env python

"""
Author: David Quail, January, 2018.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""

from GVF import *
from CycleWorld import *
from BehaviorPolicy import *

import numpy

alpha = 0.1
#numberOfActiveFeatures = 5 #1 color bit + 1 bias bit + ~2 random bits + 1 GVF bit
numberOfActiveFeatures = 2

def oneStepGammaFunction(colorObservation):
    return 0

def makeSeeColorCumulantFunction(color):
    def cumulantFunuction(colorObservation):
        val = 0
        if color == 'r':
            if colorObservation.X[0] == 1:
                val = 1
        elif color == 'g':
            if colorObservation.X[1] == 1:
                val = 1
        elif color == 'b':
            if colorObservation.X[2] == 1:
                val = 1
        elif color == 'w':
            if colorObservation.X[3] == 1:
                val = 1
        return val

    return cumulantFunuction

def makeSeeColorGammaFunction(color):
    def gammaFunction(colorObservation):
        val = 0
        if color == 'r':
            if colorObservation.X[0] == 1:
                val = 1
        elif color == 'g':
            if colorObservation.X[1] == 1:
                val = 1
        elif color == 'b':
            if colorObservation.X[2] == 1:
                val = 1
        elif color == 'w':
            if colorObservation.X[3] == 1:
                val = 1
        return val

    return gammaFunction

def zeroGamma(state):
    return 0

def timestepCumulant(state):
    return 1

def turnLeftPolicy(state):
    return "L"

def turnRightPolicy(state):
    return "R"


class LearningForeground:

    def __init__(self):
        self.cycleWorld = CycleWorld()


        self.behaviorPolicy = BehaviorPolicy()

        self.lastAction = 0
        self.currentAction = 0 #Bit of a hack to allow state representations based on GVFs to peak at last action and current action
        # self.featureRepresentationLength = 6*6*4 + 6 #6 by 6 grid, 4 orientations, + 6 color bits
        self.featureRepresentationLength = 4 + 1 + 4 + 4# ie.4 color bits + bias bit + 4 random bits + 4 GVF bits
        #Initialize the demons appropriately depending on what test you are runnning by commenting / uncommenting
        self.demons = self.createGVFs()

        self.previousState = False

    def createSameWhiteGVFs(self):
        gvfs = []
        gvf1 =  GVF(self.featureRepresentationLength,
                    alpha / numberOfActiveFeatures, isOffPolicy=False, name="1 step white")
        gvf1.gamma = oneStepGammaFunction
        gvf1.cumulant = makeSeeColorCumulantFunction('w')
        gvfs.append(gvf1)

        gvf2 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="1 step white")
        gvf2.gamma = oneStepGammaFunction
        gvf2.cumulant = makeSeeColorCumulantFunction('w')
        gvfs.append(gvf2)

        gvf3 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="1 step white")
        gvf3.gamma = oneStepGammaFunction
        gvf3.cumulant = makeSeeColorCumulantFunction('w')
        gvfs.append(gvf3)

        gvf4 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="1 step white")
        gvf4.gamma = oneStepGammaFunction
        gvf4.cumulant = makeSeeColorCumulantFunction('w')
        gvfs.append(gvf4)

        return gvfs

    def createGreenGVFs(self):
        gvfs = []
        gvf1 =  GVF(self.featureRepresentationLength,
                    alpha / numberOfActiveFeatures, isOffPolicy=False, name="1 step green")
        gvf1.gamma = oneStepGammaFunction
        gvf1.cumulant = makeSeeColorCumulantFunction('g')
        gvfs.append(gvf1)

        gvf2 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="2 step green")
        gvf2.gamma = oneStepGammaFunction
        def cumulant2(state):
            return gvf1.prediction(state)
        gvf2.cumulant = cumulant2
        gvfs.append(gvf2)

        gvf3 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="3 step green")
        gvf3.gamma = oneStepGammaFunction
        def cumulant3(state):
            return gvf2.prediction(state)
        gvf3.cumulant = cumulant3
        gvfs.append(gvf3)

        gvf4 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="4 step green")
        gvf4.gamma = oneStepGammaFunction
        def cumulant4(state):
            return gvf3.prediction(state)
        gvf4.cumulant = cumulant4
        gvfs.append(gvf4)

        return gvfs

    def createWhiteGVFs(self):
        gvfs = []
        gvf1 =  GVF(self.featureRepresentationLength,
                    alpha / numberOfActiveFeatures, isOffPolicy=False, name="1 step white")
        gvf1.gamma = oneStepGammaFunction
        gvf1.cumulant = makeSeeColorCumulantFunction('w')
        gvfs.append(gvf1)

        gvf2 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="2 step white")
        gvf2.gamma = oneStepGammaFunction
        def cumulant2(state):
            return gvf1.prediction(state)
        gvf2.cumulant = cumulant2
        gvfs.append(gvf2)

        gvf3 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="3 step white")
        gvf3.gamma = oneStepGammaFunction
        def cumulant3(state):
            return gvf2.prediction(state)
        gvf3.cumulant = cumulant3
        gvfs.append(gvf3)

        gvf4 = GVF(self.featureRepresentationLength,
                   alpha / numberOfActiveFeatures, isOffPolicy=False, name="4 step white")
        gvf4.gamma = oneStepGammaFunction
        def cumulant4(state):
            return gvf3.prediction(state)
        gvf4.cumulant = cumulant4
        gvfs.append(gvf4)

        return gvfs


    def createGVFs(self):
        return self.createSameWhiteGVFs()

    """
    Create a feature representation using the existing GVFs, history and immediate observation
    """
    def createFeatureRepresentation(self, observation):
        return self.createPartiallyObservableRepresentation(observation)



    """
    Returns information which includes:
    - the previous GVF outputs for the previous state (8 GVFs X 3 possible actions)
    - Bit corresponding to the color observed
    - Bit corresponding to the bump sensor
    """
    def createPartiallyObservableRepresentation(self, observation):
        vectorSize = len(observation) + len(self.demons)

        gvfPredictionVector = []

        for demon in self.demons:
            priorDemonBit = 0
            if self.lastAction:
                priorDemonOutput = demon.prediction(self.previousState)
                priorDemonBit = int(round(priorDemonOutput))
            gvfPredictionVector.append(priorDemonBit)
        rep = numpy.append(observation, gvfPredictionVector)
        return rep

    def start(self):

        print("Initial world:")
        self.cycleWorld.printWorld()
        i = 0
        while (True):

            if i %5000 == 0:
                print("========== Timestep: " + str(i))
            i = i + 1
            action = self.behaviorPolicy.policy(self.previousState)
            self.currentAction = action
            print("Cycle world before action: ")
            self.cycleWorld.printWorld()
            print("Action being taken: " + str(action))
            (reward, observation) = self.cycleWorld.takeAction(action)
            featureRep = self.createFeatureRepresentation(observation)
            stateRepresentation = StateRepresentation()
            stateRepresentation.X = featureRep

            self.updateDemons(self.previousState, action, stateRepresentation)
            if not self.previousState:
                self.previousState = StateRepresentation()
            self.lastAction = action

            self.previousState = stateRepresentation


    def updateDemons(self, oldState, action, newState):

        if self.previousState:
            #Learning
            #TODO - REmove after testing
            demon = self.demons[0]
            predBefore = demon.prediction(self.previousState)
            demon.learn(oldState, action, newState)
            print("Demon " + demon.name + " prediction before: " + str(predBefore))
            print("Demon" + demon.name + " prediction after: " + str(demon.prediction(self.previousState)))
            """
            for demon in self.demons:
                predBefore = demon.prediction(self.previousState)
                demon.learn(oldState, action, newState)
                print("Demon " + demon.name + " prediction before: " + str(predBefore))
                print("Demon" + demon.name + " prediction after: " + str(demon.prediction(self.previousState)))
            """



def start():
    foreground = LearningForeground()
    foreground.start()

start()