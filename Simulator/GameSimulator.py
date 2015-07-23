import numpy as np
import logging


class GameSimulator(object):  
    def __init__(self, gammeCommand, vectorCardinality, mmTreeDepth, numPlaysPerOpponent):    
        self._logger = logging.getLogger('OptimizationManager.GameSimulator')
        self._logger.debug('Initializing game simulator...')
        self._logger.debug('_gameCmd = ' + str(gammeCommand))
        self._logger.debug('_vectorCardinality = ' + str(vectorCardinality))
        self._logger.debug('_mmTreeDepth = ' + str(mmTreeDepth))
        self._logger.debug('_numPlayoutsPerOpponent = ' + str(numPlaysPerOpponent))
        
        self._gameCmd = gammeCommand
        self._vectorCardinality = vectorCardinality
        self._mmTreeDepth = mmTreeDepth
        self._numPlayoutsPerOpponent = numPlaysPerOpponent
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_logger']
        return d

    def __setstate__(self, d):
        logger = logging.getLogger('OptimizationManager.GameSimulator')
        d['_logger'] = logger
        self.__dict__.update(d)
    
    def generateOpponents(self, numOfOpponents):
        self._logger.debug('Generating ' + str(numOfOpponents) + ' players...')        
        opponents = np.zeros((numOfOpponents, self._vectorCardinality))
        
        for i in range(numOfOpponents):
            opponents[i, :] = self.generateRandomHeuristics()
        
        self._logger.debug('Generated players: ' + str(opponents))        
        return opponents
    
    def generateRandomHeuristics(self):
        randomVector = 2 * np.random.rand(1, self._vectorCardinality) - 1
        randomVector = randomVector / np.linalg.norm(randomVector)
        
        self._logger.debug('Generated random player vector ' + str(randomVector))
        return randomVector
        
    def buildPoolOfPlayers(self, n):
        pass
    
    def runGameSimulations(self, player, opponents, doSmoothing):
        pass
    
    def runGameSimulationsWithExtendedScoringReport(self, player, opponents, oppIDStrengthPairs):
        pass        
    
    def runGameSimulationsWithELORating(self, player, opponents, oppIDStrengthPairs, assumedELORating):
        pass
    
    def getHeuVectorLength(self):
        return self._vectorCardinality
    
    def dim(self):
        return self._vectorCardinality
    
    def repetetions(self):
        return self._numPlayoutsPerOpponent
    
    def numCallEvals(self, mode, numContenders):
        if mode == 'mle-tournament':
            return numContenders * (numContenders - 1) * self._numPlayoutsPerOpponent
        elif mode == 'dbgd':
            return 2 * self._numPlayoutsPerOpponent
        return 2 * numContenders * self._numPlayoutsPerOpponent
    
    def compareHeuristics(self, heuDictionary, numRounds=10, numOpponentsPerRound=50):
        scoreDict = {}        
        for i in range(numRounds):
            self._logger.info('Round ' + str(i + 1))
            
            opponents = self.generateOpponents(numOpponentsPerRound)
            self._logger.info('Fixed set of ' + str(numOpponentsPerRound) + ' random players generated (they will be used against all heuristics in this round)!')
            
            for key in heuDictionary.keys():
                entry = heuDictionary.get(key)
                score = self.runGameSimulations(entry, opponents, False)
                scoreDict.update({key : float(score) + scoreDict.get(key, 0)})
            
            self._logger.debug('Dictionary state at the of the round ' + str(scoreDict))
        
        self._logger.info('Accumulated scores: ' + str(scoreDict))
        return scoreDict
    
    def simulateMatch(self, home, away, doPlayoutSmoothing=False):
        return self.runGameSimulations(home, away.reshape(1, -1), doPlayoutSmoothing)
    
    def calculateScoreDeviation(self, numberOfEvals=2, numOpponentsPerRound=50):
        source = self.generateRandomHeuristics()
        
        opponents = self.generateOpponents(numOpponentsPerRound)
        observations = np.zeros(numberOfEvals)
        
        for i in range(numberOfEvals):
            observations[i] = self.runGameSimulations(source, opponents, False)

        scoreDeviation = np.std(observations)
        
        self._logger.debug('Calculated score deviation ' + str(scoreDeviation) + ' using ' + str(numberOfEvals) + ' oracle calls and ' + str(numOpponentsPerRound) + ' opponents per call!')        
        
        if scoreDeviation == 0:
            self._logger.debug('Non-discriminative data pools generated, using default score deviation ' + str(0.05))
            scoreDeviation = 0.05
        
        return scoreDeviation
    
    def gameName(self):
        pass
