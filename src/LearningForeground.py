#!/usr/bin/env python

"""
Author: David Quail, January, 2018.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""

from GVF import *
from CycleWorld import *
from TriggerWorld import *
from BehaviorPolicy import *
from DoubleQ import *

import numpy

alpha = 0.1
#numberOfActiveFeatures = 5 #1 color bit + 1 bias bit + ~2 random bits + 1 GVF bit
numberOfActiveFeatures = 2

def oneStepGammaFunction(colorObservation):
    return 0

def makeSeeColorCumulantFunction(color):
    def cumulantFunuction(colorObservation):
        val = 0
        if colorObservation.X == None:
            #Terminal state
            return 0
        else:
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

def makeEchoColorGammaFunction(color):
    def gammaFunction(colorObservation):
        val = 0.8
        if colorObservation.X == None:
            return 0.8
        else:
            if color == 'r':
                if colorObservation.X[0] == 1:
                    val = 0
            elif color == 'g':
                if colorObservation.X[1] == 1:
                    val = 0
            elif color == 'b':
                if colorObservation.X[2] == 1:
                    val = 0
            elif color == 'w':
                if colorObservation.X[3] == 1:
                    val = 0
            return val

    return gammaFunction

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

def moveForwardPolicy(state):
    return 'M'

class LearningForeground:

    def __init__(self):
        self.triggerWorld = TriggerWorld()

        self.behaviorPolicy = BehaviorPolicy()

        self.lastAction = 0
        self.currentAction = 0 #Bit of a hack to allow state representations based on GVFs to peak at last action and current action
        # self.featureRepresentationLength = 6*6*4 + 6 #6 by 6 grid, 4 orientations, + 6 color bits
        #self.featureRepresentationLength = 4 + 1 + 4 + 4# ie.4 color bits + bias bit + 4 random bits + 4 GVF bits
        self.featureRepresentationLength = (4 + 1 + 4 + 11) * 2  # ie.4 color bits + bias bit + 4 random bits + 11 GVF bits X 2 previous actions
        #Initialize the demons appropriately depending on what test you are runnning by commenting / uncommenting
        self.demons = self.createGVFs()

        self.doubleQ = DoubleQ(alpha=0.1, eps = 0.1, numberOfFeatures = self.featureRepresentationLength, numberOfActions = 2)

        self.previousState = False

    def createEchoGVF(self):
        gvfs = []
        gvf =  GVF(self.featureRepresentationLength,
                    alpha / numberOfActiveFeatures, isOffPolicy=True, name="Echo to green")
        gvf.gamma = makeEchoColorGammaFunction('g')
        gvf.policy = moveForwardPolicy
        gvf.cumulant = makeSeeColorCumulantFunction('g')
        gvfs.append(gvf)
        return gvfs

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
        #return self.createSameWhiteGVFs()
        return self.createEchoGVF()

    """
    Create a feature representation using the existing GVFs, history and immediate observation
    """
    def createFeatureRepresentation(self, observation, action):
        #return self.createPartiallyObservableRepresentation(observation)
        return self.createEchoRepresentation(observation, action)

    def createEchoRepresentation(self, observation, action):
        if observation == None:
            return None
        else:
            echoVector = numpy.zeros(11)
            if self.previousState:
            #if False:
                echoGVF = self.demons[0]
                echoValue = echoGVF.prediction(self.previousState)
                echoIndex = int(round(echoValue * 10))
                echoVector = numpy.zeros(11)
                echoVector[echoIndex] = 1


            rep = numpy.append(observation, echoVector)
            emptyRep = numpy.zeros(self.featureRepresentationLength / 2)

            repWithLastAction = []

            if action == "M":
                repWithLastAction = numpy.append(rep, emptyRep)
            else:
                repWithLastAction = numpy.append(emptyRep, rep)

            return repWithLastAction


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

    def resetEnvironment(self):
        self.lastAction = 0
        self.currentAction = 0
        for demon in self.demons:
            demon.reset()
            self.triggerWorld.reset()

    def start(self, numberOfEpisodes = 100, numberOfRuns = 1):

        print("Initial world:")
        self.triggerWorld.printWorld()
        episodeLengthArray = numpy.zeros(numberOfEpisodes)
        for run in range(numberOfRuns):
            print("+++++++++ Run number " + str(run + 1) + "++++++++++++")
            self.doubleQ.resetQ()
            self.resetEnvironment()

            for episode in range(numberOfEpisodes):
                print("---- Episode ---- " + str(episode))
                self.triggerWorld.reset()
                isTerminal = False
                self.lastAction = 0
                self.currentAction = 0
                self.previousState = False
                step = 0
                while not isTerminal:
                    step = step + 1
                    if step %200000 == 0:
                        print("========== Timestep: " + str(step))

                    #action = self.behaviorPolicy.policy(self.previousState)
                    if self.previousState:
                        action = self.doubleQ.policy(self.previousState.X)
                    else:
                        action = 0
                    if action == 0:
                        action = "M"
                    elif action == 1:
                        action = "T"

                    self.currentAction = action
                    print("")
                    print("------")
                    print("State learning about:")
                    if self.previousState:
                        print("X: " + str(self.previousState.X))
                    self.triggerWorld.printWorld()

                    if action == "T":
                        print("--- Trigger ---")

                    (reward, observation) = self.triggerWorld.takeAction(action)
                    if observation == None:
                        isTerminal = True
                    else:
                        isTerminal = False

                    featureRep = self.createFeatureRepresentation(observation, action)
                    stateRepresentation = StateRepresentation()
                    stateRepresentation.X = featureRep
                    if (action == "M"):
                        a = 0
                    if (action =="T"):
                        a = 1
                    if self.previousState:
                        self.doubleQ.learn(self.previousState.X, a, stateRepresentation.X, reward)
                    if not featureRep == None:
                        """
                        Kind of a temporary hack. We could normally learn from transitions to terminal states.
                        But in our case, terminal states only occer from trigger action. So nothing to learn from.
                        And our GTD algorithm breaks with terminal states
                        """
                        self.updateDemons(self.previousState, action, stateRepresentation)
                    if not self.previousState:
                        self.previousState = StateRepresentation()
                    self.lastAction = action

                    self.previousState = stateRepresentation

                    if isTerminal:
                        print("")
                        print("-- Episode finished in " + str(step) + " steps.")
                        print("")
                        print("-- Old episode length: " + str(episodeLengthArray[episode]))
                        print("- steps this time: " + str(step))
                        episodeLengthArray[episode] = episodeLengthArray[episode] + (1.0 / (run + 1.0)) * (step - episodeLengthArray[episode])
                        print("-- Adjusted episode length: " + str(episodeLengthArray[episode]))
        print("")
        print("------ finished with all runs ------ ")
        self.plotAverageEpisodeLengths(episodeLengthArray, numberOfRuns)

    def plotAverageEpisodeLengths(self, plotLengthsArray, numberOfRuns):
        fig = plt.figure(1)
        fig.suptitle('Average Episode Length', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(211)
        titleLabel = "Average over " + str(numberOfRuns) + " runs"
        ax.set_title(titleLabel)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average episode length')

        ax.plot(plotLengthsArray)

        plt.show()

    def updateDemons(self, oldState, action, newState):

        if self.previousState:
            #Learning
            #TODO - REmove after testing
            demon = self.demons[0]
            predBefore = demon.prediction(self.previousState)
            demon.learn(oldState, action, newState)
            print("Demon " + demon.name + " previous state prediction before learning: " + str(predBefore))

            print("Demon" + demon.name + " previous state prediction after learning: " + str(demon.prediction(self.previousState)))
            """
            for demon in self.demons:
                predBefore = demon.prediction(self.previousState)
                demon.learn(oldState, action, newState)
                print("Demon " + demon.name + " prediction before: " + str(predBefore))
                print("Demon" + demon.name + " prediction after: " + str(demon.prediction(self.previousState)))
            """



def start():
    foreground = LearningForeground()
    foreground.start(numberOfEpisodes =  1000, numberOfRuns = 50)

start()
