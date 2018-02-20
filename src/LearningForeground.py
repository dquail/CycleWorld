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
from random import randint
import numpy
from pylab import *

alpha = 0.1
#numberOfActiveFeatures = 5 #1 color bit + 1 bias bit + ~2 random bits + 1 GVF bit
numberOfActiveFeatures = 3

def makeSeeBitCumulantFunction(bitIndex):
    def cumulantFunuction(colorObservation):
        val = 0
        if colorObservation.X is None:
            #Terminal state
            return 0
        else:
            if colorObservation.X[bitIndex] == 1:
                val = 1
            return val
    return cumulantFunuction

def makeEchoBitGammaFunction(bitIndex):
    def gammaFunction(colorObservation):
        val = 0.8
        if colorObservation.X is None:
            return 0.8
        else:
            if colorObservation.X[bitIndex] == 1:
                val = 0
            return val

    return gammaFunction

def moveForwardPolicy(state):
    return 'M'

class LearningForeground:

    def __init__(self):
        self.triggerWorld = TriggerWorld()
        self.epsilon = 0.1
        self.behaviorPolicy = BehaviorPolicy()
        self.candidateBits = list(range(9))
        self.bitsPredicted = []
        self.lastAction = 0
        self.currentAction = 0 #Bit of a hack to allow state representations based on GVFs to peak at last action and current action
        # self.featureRepresentationLength = 6*6*4 + 6 #6 by 6 grid, 4 orientations, + 6 color bits
        #self.featureRepresentationLength = 4 + 1 + 4 + 4# ie.4 color bits + bias bit + 4 random bits + 4 GVF bits
        #self.featureRepresentationLength = (4 + 1 + 4 + 11) * 2  # ie.4 color bits + bias bit + 4 random bits + 11 GVF bits X 2 previous actions
        self.featureRepresentationLength = (4 + 1 + 4 + 11 * 2) * 2  # ie.4 color bits + bias bit + 4 random bits + 11X2 GVF bits X 2 previous actions
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


    def createGVFs(self):
        #return self.createSameWhiteGVFs()
        #return self.createEchoGVF()
        return self.initializeRandomGVFS()

    """
    Create a feature representation using the existing GVFs, history and immediate observation
    """
    def createFeatureRepresentation(self, observation, action):
        #return self.createPartiallyObservableRepresentation(observation)
        #return self.createEchoRepresentation(observation, action)
        return self.createRepresentationWithGVFs(observation, action)

    def createRepresentationWithGVFs(self, observation, action):
        if observation is None:
            return None
        else:
            predictiveBits = []
            for echoGVF in self.demons:
                echoVector = numpy.zeros(11)
                if self.previousState:
                    echoValue = echoGVF.prediction(self.previousState)
                    if echoValue > 0.0:
                        echoIndex = int(round(echoValue * 10))
                        if echoIndex > 10:
                            echoIndex = 10
                        echoVector = numpy.zeros(11)
                        echoVector[echoIndex] = 1

                predictiveBits = numpy.append(predictiveBits, echoVector)

            rep = numpy.append(observation, predictiveBits)
            emptyRep = numpy.zeros(int(self.featureRepresentationLength / 2))

            repWithLastAction = []

            if action == "M":
                repWithLastAction = numpy.append(rep, emptyRep)
            else:
                repWithLastAction = numpy.append(emptyRep, rep)

            return repWithLastAction


    def resetEnvironment(self):
        self.lastAction = 0
        self.currentAction = 0
        self.demons = self.initializeRandomGVFS()
        """
        for demon in self.demons:
            demon.reset()
        """
        self.candidateBits = list(range(9))
        self.bitsPredicted = []
        self.triggerWorld.reset()

    def start(self, numberOfEpisodes = 100, numberOfRuns = 1):

        print("Initial world:")
        self.triggerWorld.printWorld()
        episodeLengthArray = numpy.zeros(numberOfEpisodes)
        for run in range(numberOfRuns):
            print("RUN NUMER " + str(run + 1) + " .............")

            self.doubleQ.resetQ()
            self.resetEnvironment()

            for episode in range(numberOfEpisodes):

                if episode %2500 == 0:
                    #print("XXXX Episode " + str(episode) + " kulling demons XXXX")
                    self.kullDemon()

                if episode %1000 == 0:
                    print("--- Episode " + str(episode) + " ... ")
                self.triggerWorld.reset()
                isTerminal = False
                self.lastAction = 0
                self.currentAction = 0
                self.previousState = False
                step = 0
                while not isTerminal:
                    step = step + 1

                    #action = self.behaviorPolicy.policy(self.previousState)
                    if self.previousState:
                        randomE = random()
                        if (randomE < self.epsilon):
                            #explore
                            action = randint(0,1)
                        else:
                            action = self.doubleQ.policy(self.previousState.X)
                    else:
                        action = 0
                    if action == 0:
                        action = "M"
                    elif action == 1:
                        action = "T"

                    #TODO - Remove after testing
                    """
                    if episode == 0:
                        if (step < 10000):
                            action = "M"
                        else:
                            print("OK now trying")
                    """
                    self.currentAction = action


                    """
                    print("State: ")
                    self.triggerWorld.printWorld()
                    print("- Action decision: " + str(action))
                    print(self.demons[1].name)
                    if (self.previousState):
                        prediction = self.demons[1].prediction(self.previousState)
                        print("Prediction: " + str(prediction))
                    """

                    (reward, observation) = self.triggerWorld.takeAction(action)
                    if observation is None:
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
                    if not featureRep is None:
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
                        """
                        print("")
                        print("-- Episode finished in " + str(step) + " steps.")
                        print("")
                        print("-- Old episode length: " + str(episodeLengthArray[episode]))
                        print("- steps this time: " + str(step))
                        print("-- Adjusted episode length: " + str(episodeLengthArray[episode]))
                        """
                        print("Steps in episode  " + str(episode) + ": " + str(step))
                        if episode > 0:
                            episodeLengthArray[episode] = episodeLengthArray[episode] + (1.0 / (run + 1.0)) * (
                                    step - episodeLengthArray[episode])
            input("Finished run. Press ENTER to continue ...")
        self.plotAverageEpisodeLengths(episodeLengthArray, numberOfRuns)

    def plotAverageEpisodeLengths(self, plotLengthsArray, numberOfRuns):
        b = plotLengthsArray[len(plotLengthsArray) - 10: len(plotLengthsArray) - 10 + 5]
        fig = plt.figure(1)
        fig.suptitle('Average Episode Length', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(211)
        titleLabel = "Average over " + str(numberOfRuns) + " runs"
        ax.set_title(titleLabel)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average episode length')

        ax.plot(plotLengthsArray)

        plt.show()

    def randomBitIndex(self, excludeBitsTried = False):
        randomBit = 0
        if not excludeBitsTried:
            randomBit = numpy.random.randint(0, 9)
        else:
            #Naive strategy. If there are no candidate bits left to choose from, repopulate them all
            if (len(self.candidateBits) == 0):
                self.candidateBits = list(range(9))
                for bit in self.bitsPredicted:
                    self.candidateBits.remove(bit)
            randomBit = numpy.random.choice(self.candidateBits)

        self.candidateBits.remove(randomBit)
        self.bitsPredicted.append(randomBit)

        return randomBit
        #exclusions is an array of indexes to not choose from

    def createRandomGVF(self):
        randBit = randint(0, 9) #TODO - make this actually random again.
        #randBit = randint(0, 3)  # TODO - make this actually random again.
        if randBit == 1:
            print("!!!!!!!!!!! Got it !!!!!!!!!!!!!!!")
            input("Replaced with proper GVF. ENTER to continue ...")
        print("Creating GVF with bit: " + str(randBit))
        gvf = GVF(self.featureRepresentationLength,
                  alpha / numberOfActiveFeatures, isOffPolicy=True, name="Echo to bit " + str(randBit))
        gvf.gamma = makeEchoBitGammaFunction(randBit)
        gvf.policy = moveForwardPolicy
        gvf.cumulant = makeSeeBitCumulantFunction(randBit)
        return gvf

    def initializeRandomGVFS(self):
        #Pick a bit in the observation to see it's echo value. ie. approximate how long until you see it.

        gvfs = []
        for i in range(2):

            randBit = randint(0, 9) #TODO - make this actually random again.
            #randBit = randint(0, 3)  # TODO - make this actually random again.

            if randBit == 1:
                input("Created GVF for input[1] in first shot. ENTER to continue ...")
                print("!!!!!!!!!!!!!!! Got it !!!!!!!!!!!!!!!")


            #randBit = i
            #randBit = i + 5

            gvf =  GVF(self.featureRepresentationLength,
                        alpha / numberOfActiveFeatures, isOffPolicy=True, name="Echo to bit " + str(randBit))
            gvf.gamma = makeEchoBitGammaFunction(randBit)
            gvf.policy = moveForwardPolicy
            gvf.cumulant = makeSeeBitCumulantFunction(randBit)
            print("New gvf: " + str(gvf.name))
            gvfs.append(gvf)
        return gvfs

    def kullDemon(self):
        #Determine weakest GVF and replace it with a random one.
        indexOffset = 31
        gvfOneScore = 0
        #Look at the appropriate indexes in self.doubleQ to sum up the abs score
        gvfOneIndexes = []
        i = list(range(9,20))
        gvfOneIndexes.extend(i)
        i = list(range(9 + indexOffset * 1, 20 + indexOffset * 1))
        gvfOneIndexes.extend(i)
        i = list(range(9 + indexOffset * 2, 20 + indexOffset * 2))
        gvfOneIndexes.extend(i)
        i = list(range(9 + indexOffset * 3, 20 + indexOffset * 3))
        gvfOneIndexes.extend(i)
        for index in gvfOneIndexes:
            gvfOneScore += fabs((self.doubleQ.theta1[index] + self.doubleQ.theta2[index]) / 2.0)

        gvfTwoIndexes = []
        i = list(range(21, 31))
        gvfTwoIndexes.extend(i)
        i = list(range(21 + indexOffset * 1, 31 + indexOffset * 1))
        gvfTwoIndexes.extend(i)
        i = list(range(21 + indexOffset * 2, 31 + indexOffset * 2))
        gvfTwoIndexes.extend(i)
        i = list(range(21 + indexOffset * 3, 31 + indexOffset * 3))
        gvfTwoIndexes.extend(i)

        gvfTwoScore = 0
        for index in gvfTwoIndexes:
            gvfTwoScore += fabs((self.doubleQ.theta1[index] + self.doubleQ.theta2[index]) / 2.0)
        gvfNew = self.createRandomGVF()

        print("New GVF: " + str(gvfNew.name))

        if gvfOneScore > gvfTwoScore:
            #Recycle gvfTwoScore
            print("Kulled " + str(self.demons[1].name))
            if (self.demons[1].name == "Echo to bit 1"):
                input("Kulling wrong one. Enter to continue ...")
            self.demons[1] = gvfNew

        else:
            #recycle gvf 1
            print("Kulled " + str(self.demons[0].name))
            if (self.demons[0].name == "Echo to bit 1"):
                input("Kulling wrong one. Enter to continue ...")
            self.demons[0] = gvfNew


    def updateDemons(self, oldState, action, newState):

        if self.previousState:
            #Learning

            for demon in self.demons:
                predBefore = demon.prediction(self.previousState)
                demon.learn(oldState, action, newState)
                #print("Demon " + demon.name + " prediction before: " + str(predBefore))
                #print("Demon" + demon.name + " prediction after: " + str(demon.prediction(self.previousState)))




def start():
    foreground = LearningForeground()
    foreground.start(numberOfEpisodes =  20000, numberOfRuns = 100)

start()
