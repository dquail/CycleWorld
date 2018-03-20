#!/usr/bin/env python

"""
Author: David Quail, January, 2018.

Description:
LearningForeground contains a collection of GVF's. It accepts new state representations, learns, and then takes action.

"""

from GVF import *
from TriggerWorld import *
from BehaviorPolicy import *
from DoubleQ import *
from random import randint
import numpy
from pylab import *

alpha = 0.1
#numberOfActiveFeatures = 5 #1 color bit + 1 bias bit + ~2 random bits + 1 GVF bit
numberOfActiveFeatures = 2

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
        self.correctActionArray = []
        self.triggerWorld = TriggerWorld()
        self.epsilon = 0.2
        self.behaviorPolicy = BehaviorPolicy()
        self.lastAction = 0
        self.gotBit1 = False
        self.doKullRate = 0
        self.currentAction = 0 #Bit of a hack to allow state representations based on GVFs to peak at last action and current action
        # self.featureRepresentationLength = 6*6*4 + 6 #6 by 6 grid, 4 orientations, + 6 color bits
        #self.featureRepresentationLength = 4 + 1 + 4 + 4# ie.4 color bits + bias bit + 4 random bits + 4 GVF bits
        #self.featureRepresentationLength = (4 + 1 + 4 + 11) * 2  # ie.4 color bits + bias bit + 4 random bits + 11 GVF bits X 2 previous actions
        self.featureRepresentationLength = (4 + 1 + 4 + 11 * 2) * 2  # ie.4 color bits + bias bit + 4 random bits + 11X2 GVF bits X 2 previous actions
        #Initialize the demons appropriately depending on what test you are runnning by commenting / uncommenting
        self.actionInRun = 0

        self.candidateGVFs  = []
        self.kulledGVFs = []
        self.activeGVFs = []

        self.createGVFs()

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
        self.resetGVFS()

    """
    Create a feature representation using the existing GVFs, history and immediate observation
    """
    def createFeatureRepresentation(self, observation, action):
        #return self.createPartiallyObservableRepresentation(observation)
        #return self.createEchoRepresentation(observation, action)
        return self.createRepresentationWithGVFs(observation, action)
        #return self.createFullyObservableRepresentation(observation, action)


    def createFullyObservableRepresentation(self, observation, action):
        if observation is None:
            return None
        else:
            predictiveBits = numpy.zeros(22)

            rep = numpy.append(observation, predictiveBits)
            emptyRep = numpy.zeros(int(self.featureRepresentationLength / 2))

            repWithLastAction = []

            if action == "M":
                repWithLastAction = numpy.append(rep, emptyRep)
            else:
                repWithLastAction = numpy.append(emptyRep, rep)

            return repWithLastAction

    def createRepresentationWithGVFs(self, observation, action):
        if observation is None:
            return None
        else:
            predictiveBits = []
            for echoGVF in self.activeGVFs:
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
        #self.demons = self.initializeRandomGVFS()
        self.resetGVFS()
        self.doubleQ.resetQ()
        """
        for demon in self.demons:
            demon.reset()
        """
        self.actionInRun = 0
        self.triggerWorld.reset()


    def weakestGVF(self):
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

        gvfZeroIndexes = []
        i = list(range(21, 31))
        gvfZeroIndexes.extend(i)
        i = list(range(21 + indexOffset * 1, 31 + indexOffset * 1))
        gvfZeroIndexes.extend(i)
        i = list(range(21 + indexOffset * 2, 31 + indexOffset * 2))
        gvfZeroIndexes.extend(i)
        i = list(range(21 + indexOffset * 3, 31 + indexOffset * 3))
        gvfZeroIndexes.extend(i)

        gvfZeroScore = 0
        for index in gvfZeroIndexes:
            gvfZeroScore += fabs((self.doubleQ.theta1[index] + self.doubleQ.theta2[index]) / 2.0)

        gvfToKull = None

        if gvfOneScore > gvfZeroScore:
            #Recycle gvfTwoScore
            gvfToKull = self.activeGVFs[1]
            #Reset the weights
            for index in gvfZeroIndexes:
                self.doubleQ.theta1[index] = 0.0
                self.doubleQ.theta2[index] = 0.0

        else:
            gvfToKull = self.activeGVFs[0]
            for index in gvfOneIndexes:
                self.doubleQ.theta1[index] = 0.0
                self.doubleQ.theta2[index] = 0.0
        return gvfToKull

    def kullAndCreate(self):
        #Determine weakest GVF
        weakestGVF = self.weakestGVF()
        if weakestGVF.name == "Echo to bit 1":
            self.gotBit1 = False
            #input("Kulling wrong bit!!! ENTER to continue")
        self.kulledGVFs.append(weakestGVF)
        weakestIndex = self.activeGVFs.index(weakestGVF)

        #Get a new candidate GVF
        if len(self.candidateGVFs) == 0:
            #No more candidate GVFS to try. Move all kulled ones back (but not active
            for gvf in self.kulledGVFs:
                self.candidateGVFs.append(gvf)
            self.kulledGVFs = []
        gvf = self.getRandomGVFFromCandidates()

        if gvf.name == "Echo to bit 1":
            #input("Getting first bit. ENTER to continue ....")
            self.gotBit1 = True

        self.activeGVFs[weakestIndex] = gvf
        self.candidateGVFs.remove(gvf)

    def resetGVFS(self):
        #Pick a bit in the observation to see it's echo value. ie. approximate how long until you see it.
        self.candidateGVFs  = []
        self.kulledGVFs = []
        self.activeGVFs = []
        self.gotBit1 = False

        #Reinitialize the candidate GVFS
        for i in range(9):
            """
            if i == 3:
                i = 5
            """
            gvf =  GVF(self.featureRepresentationLength,
                        alpha / numberOfActiveFeatures, isOffPolicy=True, name="Echo to bit " + str(i))
            gvf.gamma = makeEchoBitGammaFunction(i)
            gvf.policy = moveForwardPolicy

            gvf.cumulant = makeSeeBitCumulantFunction(i)

            self.candidateGVFs.append(gvf)


        #Move 2 random candidates to active
        for i in range(2):
            if self.doKullRate > 0:
                gvf = self.getRandomGVFFromCandidates()
            else:
                gvf = self.candidateGVFs[i+1]
                #gvf = self.candidateGVFs[i + 4]

            if gvf.name == "Echo to bit 1":
                self.gotBit1 = True
                #input("Lucky. Getting first bit. ENTER to continue ....")
                print("Lucky. Getting first bit.")

            self.candidateGVFs.remove(gvf)
            self.activeGVFs.append(gvf)


        """
        #Can use the below to always reset with green and white bits

        greenGVF = self.candidateGVFs[1]
        whiteGVF = self.candidateGVFs[3]
        self.candidateGVFs.remove(greenGVF)
        self.activeGVFs.append(greenGVF)

        self.candidateGVFs.remove(whiteGVF)
        self.activeGVFs.append(whiteGVF)
        """

    def getRandomGVFFromCandidates(self):
        if len(self.candidateGVFs) == 0:
            return None
        else:
            randIndex = randint(0, len(self.candidateGVFs))
            return self.candidateGVFs[randIndex]


    def updateCorrectActionArray(self, correctActionTaken, run):
        """
        episodeLengthArray[episode] = episodeLengthArray[episode] + (1.0 / (run + 1.0)) * (
                                    step - episodeLengthArray[episode])
        """
        if run == 0 or len(self.correctActionArray) <= self.actionInRun:
            self.correctActionArray.append(correctActionTaken)
        else:
            self.correctActionArray[self.actionInRun] = self.correctActionArray[self.actionInRun] + (1.0 / (run + 1.0)) * (correctActionTaken - self.correctActionArray[self.actionInRun])
        self.correctActionArray

        self.actionInRun = self.actionInRun + 1

    def start(self, numberOfEpisodes = 100, numberOfRuns = 1, doKullRate = 0):
        self.doKullRate = doKullRate
        print("Initial world:")
        self.triggerWorld.printWorld()
        episodeLengthArray = numpy.zeros(numberOfEpisodes)
        for run in range(numberOfRuns):
            print("RUN NUMER " + str(run + 1) + " .............")

            self.resetEnvironment()

            for episode in range(1, numberOfEpisodes):
                if doKullRate > 0:
                    if episode % doKullRate == 0:
                        print("XXXX Episode " + str(episode) + " kulling demons XXXX")
                        print("GVFS before: ")
                        print(self.activeGVFs[0].name + ", " + self.activeGVFs[1].name)
                        self.kullAndCreate()
                        print("GVFS after: ")
                        print(self.activeGVFs[0].name + ", " + self.activeGVFs[1].name)
                        if self.gotBit1:
                            print("- Got proper bit")

                if episode %10000 == 0:
                    print("Run " + str(run) + ", Episode " + str(episode) + " ... ")
                self.triggerWorld.reset()
                isTerminal = False
                self.lastAction = 0
                self.currentAction = 0
                self.previousState = False
                step = 0
                didExploreThisEpisode = False
                while not isTerminal:
                    step = step + 1
                    didExploreThisStep = False
                    #action = self.behaviorPolicy.policy(self.previousState)
                    if self.previousState:
                        randomE = random()
                        if (randomE < self.epsilon):
                            #explore
                            didExploreThisStep = True
                            didExploreThisEpisode = True
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

                    #print("State: ")
                    #self.triggerWorld.printWorld()
                    didExploreString = ""
                    if didExploreThisStep:
                        didExploreString = " (explore) "
                    #print("- Action decision: " + str(action) + didExploreString)
                    (reward, observation, correctActionTaken) = self.triggerWorld.takeAction(action)

                    #Update statistics about taking the correct action. Only if it didn't explore and behavior
                    #policy was used (ie. previous state)
                    if not didExploreThisStep and self.previousState:
                        self.updateCorrectActionArray(correctActionTaken, run)

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
                        gotitStr = ""
                        didExploreStr = ""
                        if self.gotBit1:
                            gotitStr = " ***** With bit ****"
                        if didExploreThisEpisode:
                            didExploreStr = " --- DID explore this episode"
                        """
                        ******************************************************
                        Following line for debugging and monitoring status
                        ******************************************************
                        """
                        #print("Run: " + str(run) + ", Episode:  " + str(episode) + ", Steps: " + str(step) + str(gotitStr) + str(didExploreStr))

                        if episode > 0 and not didExploreThisEpisode:
                            episodeLengthArray[episode] = episodeLengthArray[episode] + (1.0 / (run + 1.0)) * (
                                    step - episodeLengthArray[episode])
            #input("Finished run. Press ENTER to continue ...")
        self.plotAverageEpisodeLengths(episodeLengthArray, numberOfRuns)
        #self.plotAverageCorrectDecisions(numberOfRuns)

    def plotAverageCorrectDecisions(self, numberOfRuns):
        newArray = []
        i = 0
        for a in self.correctActionArray:
            if i %50 == 0:
                newArray.append(a)
            i = i+1
        fig = plt.figure(1)
        fig.suptitle('Average correct decision', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(211)
        titleLabel = "Average over " + str(numberOfRuns) + " runs"
        ax.set_title(titleLabel)
        ax.set_xlabel('Step')
        ax.set_ylabel('Average correct decision %')

        ax.plot(newArray)

        plt.show()

    def plotAverageEpisodeLengths(self, plotLengthsArray, numberOfRuns):

        fig = plt.figure(1)
        fig.suptitle('Average Episode Length', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(211)
        #axes = plt.gca()

        ax.set_ylim([3, 15])
        titleLabel = "Average over " + str(numberOfRuns) + " runs"
        ax.set_title(titleLabel)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average episode length')

        ax.plot(plotLengthsArray)

        plt.show()

    def updateDemons(self, oldState, action, newState):

        if self.previousState:
            #Learning

            for demon in self.activeGVFs:
                predBefore = demon.prediction(self.previousState)
                demon.learn(oldState, action, newState)
                #print("Demon " + demon.name + " prediction before: " + str(predBefore))
                #print("Demon" + demon.name + " prediction after: " + str(demon.prediction(self.previousState)))




def start():
    foreground = LearningForeground()
    #foreground.start(numberOfEpisodes = 300000, numberOfRuns =100, doKullRate = 25000)
    #No kulling. No proper GVFS (no learning)
    foreground.start(numberOfEpisodes=300000, numberOfRuns=100, doKullRate=0)
    #foreground.start(numberOfEpisodes=100000, numberOfRuns=1, doKullRate=0)
    #foreground.start(numberOfEpisodes=30000, numberOfRuns=1, doKullRate=0)

start()
