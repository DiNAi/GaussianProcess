import numpy as np
import subprocess as sbp
import sys
import time
import Queue
import logging
from Simulator import GameSimulator as GS

class Timeout(Exception):
    pass

class  CheckersSimulator(GS.GameSimulator):
    def __init__(self, gammeCommand, vectorCardinality, mmTreeDepth, numPlaysPerOpponent, unratedGainInfluenceFactor = 35, gainInfluenceFactor=20, gainEstimationFactor=0.0065, defaultStrengthValue=2000):
        super( CheckersSimulator, self).__init__(gammeCommand, vectorCardinality, mmTreeDepth, numPlaysPerOpponent)
        
        self._logger = logging.getLogger('OptimizationManager. CheckersSimulator')        
        self.__gainInfluence = gainInfluenceFactor
        self.__unratedInfluenceFactor = unratedGainInfluenceFactor
        self.__gainEstimationFactor = gainEstimationFactor
        self.__initialAssumedStrength = defaultStrengthValue
        self.__vectorCardinality = vectorCardinality
        self.input_data_set = np.empty((0,2*self.__vectorCardinality),float)
        self.test_data_set = np.empty((0,2*self.__vectorCardinality),float)
        self.output_data_set = np.empty((0,1),int)
        self.output_test_data = np.empty((0,1),int)
        
        self._logger.debug('__gainInfluence = ' + str(self.__gainInfluence))
        self._logger.debug('__unratedInfluenceFactor = ' + str(self.__unratedInfluenceFactor))
        self._logger.debug('__gainEstimationFactor = ' + str(self.__gainEstimationFactor))
        self._logger.debug('__initialAssumedStrength = ' + str(self.__initialAssumedStrength))    
        self._logger.debug('Othello game simulator initialized successfully...')
    
    def __setstate__(self, d):
        logger = logging.getLogger('OptimizationManager. CheckersSimulator')
        d['_logger'] = logger
        self.__dict__.update(d)
    
    def __parseStrengthValue(self, simResults, numOfOpponents, doPlayoutSmoothing=True):
        gainsByOpponents = self._parsePlayoutGains(simResults, doPlayoutSmoothing)
        parsedStrength = sum(gainsByOpponents) / numOfOpponents
        
        self._logger.debug('Parsed strength ' + str(parsedStrength))        
        return parsedStrength
        
    def _executeInParallel(self, commands, timeout=240, poll_seconds=.1): 
        procs = [sbp.Popen(command, bufsize=0, stdout=sbp.PIPE, stderr=sbp.PIPE, shell = True) for command in commands ]
        deadline = time.time() + timeout
        while time.time() < deadline and any([proc.poll() == None for proc in procs]):
            time.sleep(poll_seconds)
      
        for proc in procs:
            if proc.poll() == None:
                if float(sys.version[:3]) >= 2.6:
                    proc.kill()
        
                raise Timeout()

        results = [(proc.communicate(), proc.returncode) for proc in procs]
       
        return results
        
    def _simulate(self, player, opponents):
        results = None
        
        simTime = time.time()
        try:
            self._logger.debug('Player ' + str(player) + ' shape ' + str(player.shape))
            cmds = self.__buildSimulationCmd(player, opponents)
            timeout = 4 * self._numPlayoutsPerOpponent * len(cmds)            
            results = self._executeInParallel(cmds, timeout)
        except ValueError:
            self._logger.error('Error while running game simulations (value error)! Retrying simulation...')
            results = self._retryGameSimulation(cmds, opponents.shape[0], 2 * timeout)
        except Timeout:
            self._logger.error('Error while running game simulations (timeout reached)! Retrying simulation...')
            results = self._retryGameSimulation(cmds, opponents.shape[0], 2 * timeout)
        
        simTime = time.time() - simTime
        print "RunTime :",round(simTime,3)
        self._logger.info('Duration of simulation = ' + str(simTime))
        
        self._logger.debug('Simulation result ' + str(results))
        return results
        
    def runGameSimulations(self, heuVector, opponents, doPlayoutSmoothing=True):
        observedStrength = np.NaN     
           
        if opponents == None:
            return observedStrength;
        
        results = self._simulate(heuVector, opponents)
        return self.__parseStrengthValue(results, opponents.shape[0], doPlayoutSmoothing) if (results != None) else np.NaN
    
    def playTournament(self, players):
        n = players.shape[0]#gets the total number of players
        player1 = np.zeros((1, self.__vectorCardinality))
        player2 = np.zeros((1, self.__vectorCardinality))
        for i in range(n):
            for j in range(n):
                if j <= i:
                    continue # A heuristic shouldn't play with itself
                player1[0, :] = players[i]#Some correction which makes the game comaptible to process a game given multiple heuristics
                player2[0, :] = players[j]
                print "Player ",i, "Vs Player ",j 
                s = self._simulate(player1, player2) # simulate a game 
                result = self._calcIndividualGains(s)                
                self.updateDataSet( player1, player2, result[0] )#updates the data set
        return None

    def benchMarkHeuristic(self, pivot, players):
        temp = self._numPlayoutsPerOpponent
        self._numPlayoutsPerOpponent = 1
        wins = loses = index = draws = 0
        player1 = np.zeros((1, self.__vectorCardinality))
        player1[0, :] = pivot[0]
        player2 = np.zeros((1, self.__vectorCardinality))
        for player in players:
            player2[0, :] = player
            print "Game Number:", index
            s = self._simulate(player1, player2) # simulate a game 
            result = self._calcIndividualGains(s)   
            res = int(result[0][0])
            if res == 1:
                wins += 1
            elif res == -1:
                loses += 1
            else:
                draws += 1
            index += 1
            
        print "wins =", wins
        
        print "loses =", loses  
        self._numPlayoutsPerOpponent = temp
        return wins+(draws * 0.5 )
    
    def playPairings(self, pivot, players):#1 Vs Many

        temp_result = self._simulate(pivot, players)
        result = self._calcIndividualGains(temp_result)
        index=0
        opponent = np.zeros((1, self.__vectorCardinality))
        for player in players:
            opponent[0, :] = player
            self.updateDataSet( pivot, opponent, result[index] )
            index = index + 1
        return None
        
    def __buildSimulationCmd(self, heuVector, opponents):  
        commands=[]  
        for opponent in opponents:        
            a='CheckersTest.exe'
            for j in heuVector:
                for i in j:
                    a = a +' ' + str(i)
                for j in opponent:
                    a = a +' ' + str(j)   
                a = a + ' ' +str(self._numPlayoutsPerOpponent)
                a = a + ' ' + str(self._mmTreeDepth)
                commands.append(a)
        return commands
        
    def _parsePlayoutGain(self, s, doSmoothing=True):
        playoutGain = float(s)
        
        self._logger.debug('Observed gain = ' + str(playoutGain))
        
        if (doSmoothing):
            playoutGain = playoutGain * self._numPlayoutsPerOpponent
                
            if (playoutGain > 0):
                playoutGain = (playoutGain * (self._numPlayoutsPerOpponent + 2)) / np.square(self._numPlayoutsPerOpponent + 1)
            else:
                playoutGain = 1.0 / np.square(self._numPlayoutsPerOpponent + 1)
        
        self._logger.debug('Parsed gain = ' + str(playoutGain))        
        return playoutGain
        
    def __parseSmoothedScoringReport(self, results, electedPairs, n):
        observedPairs = []        
        
        gainsByOpponents = self._parsePlayoutGains(results)        
        
        for i in range(len(gainsByOpponents)):
            observedPairs.append((1 - gainsByOpponents[i], electedPairs[i][1]))
    
        return sum(gainsByOpponents) / n, observedPairs
    
    def runGameSimulationsWithELORating(self, player, opponents, oppIDStrengthPairs, assumedELORating):
        n = len(opponents)      
    
        self._logger.debug('Input (strengths, opponent ID) pairs to ELO rating ' + str(oppIDStrengthPairs))    
    
        results = self._simulate(player, opponents)
        return self.__parseELOStrength(assumedELORating, results, oppIDStrengthPairs, n) if (results != None) else np.NaN
    
    def _calcIndividualGains(self, results):
        gainsByOpponents = []
        
        for (stdout, stderr), status in results:
            out = stdout.split("\n")
            out = out[0].split()
            gainsByOpponents.append(out)
        
        return gainsByOpponents
    
    def _parsePlayoutGains(self, results, doSmoothing=True):
        gainsByOpponents = []
        
        for (stdout, stderr), status in results:
            out = stdout.split("\n")
            if len(out) < 2:
                raise ValueError, 'Strength value could not be parsed!\n', str(out)
            
            summary = out[-2].split()
            gainsByOpponents.append(self._parsePlayoutGain(summary[-1], doSmoothing))
        
        self._logger.debug('Playout gains by opponents ' + str(gainsByOpponents))        
        return gainsByOpponents
    
    def __parseELOStrength(self, assumedELORating, results, oppIDStrengthPairs, n):
        gainsByOpponents = self._parsePlayoutGains(results)
        
        strength = assumedELORating
        
        opponentAdjustments = []
        accumulatedPlayerDelta = 0
        for i in range(len(gainsByOpponents)):
            estimatedPlayerGain = self.__estimateMatchGain(strength, oppIDStrengthPairs[i][0])
            
            self._logger.debug('Estimated match gain ' + str(estimatedPlayerGain) + ' against opponent with observed strength ' + str(oppIDStrengthPairs[i][0]))
            
            gainDelta = gainsByOpponents[i] - estimatedPlayerGain
            accumulatedPlayerDelta = accumulatedPlayerDelta + (gainDelta)
            ratedDelta = round(self.__gainInfluence * gainDelta) # use smaller factor for already rated players
            
            adjustedOpponentStrength = oppIDStrengthPairs[i][0] - ratedDelta 
              
            self._logger.debug('Input opponent strength = ' + str(oppIDStrengthPairs[i][0]))
            self._logger.debug('Adjusted opponent strength = ' + str(adjustedOpponentStrength))            
            
            opponentAdjustments.append((adjustedOpponentStrength, oppIDStrengthPairs[i][1]))            
        
        strength = round(strength + self.__unratedInfluenceFactor * accumulatedPlayerDelta) # gather all the deltas and adjust opponent strength
        self._logger.debug('Player strength = ' + str(strength))
        
        return strength, opponentAdjustments
        
    def buildPoolOfPlayers(self, poolSize, numberOfOpponentsPerObservation, useELOScoreMeasure=True, useFixedPoolOpponents=False):
        self._logger.debug('Building initial pool of players...')
        self._logger.debug('useELOScoreMeasure = ' + str(useELOScoreMeasure))
        self._logger.debug('poolSize = ' + str(poolSize))
        self._logger.debug('numberOfOpponentsPerObservation = ' + str(numberOfOpponentsPerObservation))
        self._logger.debug('useFixedPoolOpponents = ' + str(useFixedPoolOpponents))
        
        observedStrengths = np.zeros((1, poolSize))
        poolPlayers = self.generateOpponents(poolSize)      
                
        if (useELOScoreMeasure):
            opponentQueue = Queue.Queue(0)
            for i in range(poolSize):
                opponentQueue.put(i)
            
            for i in range(poolSize):                
                opponents = np.zeros((numberOfOpponentsPerObservation, self._vectorCardinality))
                electedIDPairs = []             
                
                for j in range(numberOfOpponentsPerObservation):
                    opponentPoolId = opponentQueue.get()
                    opponentQueue.put(opponentPoolId) # return to the queue to be used in subsequent operations
                    if opponentPoolId == i:
                        opponentPoolId = opponentQueue.get() # we will not play heuristics against themselves
                        opponentQueue.put(opponentPoolId)
                        
                    opponents[j, :] = poolPlayers[opponentPoolId, :]
                    ostrg = observedStrengths[0][opponentPoolId]
                    electedIDPairs.append((ostrg if ostrg != 0 else self.__initialAssumedStrength, opponentPoolId))
                    
                observedStrengths[0, i], adjustedPairs = self.runGameSimulationsWithELORating(poolPlayers[i, :], opponents, electedIDPairs, self.__initialAssumedStrength)
                
                # update the scores using score results from ELO rating
                for j in range(numberOfOpponentsPerObservation):
                    opponentId = electedIDPairs[j][1]                  
                    observedStrengths[0, opponentId] = adjustedPairs[j][0]
        else:
            for i in range(poolSize):
                if (i == 0 or useFixedPoolOpponents == False):
                    opponents = self.generateOpponents(numberOfOpponentsPerObservation)
                observedStrengths[0, i] = self.runGameSimulations(poolPlayers[i, :], opponents)
                
        self._logger.debug('Pool players ' + str(poolPlayers))
        self._logger.debug('Observed strengths ' + str(observedStrengths))
        return poolPlayers, observedStrengths
    
    def __estimateMatchGain(self, playerStrength, opponentStrength):
        self._logger.debug('Estimating gain for strengths with delta = ' + str(playerStrength - opponentStrength))
        return 1 / (1 + np.exp(self.__gainEstimationFactor * (opponentStrength - playerStrength)))
    
    def runGameSimulationsWithExtendedScoringReport(self, heuVector, opponents, oppIDStrengthPairs):        
        n = len(opponents)      
        results = self._simulate(heuVector, opponents)
        return self.__parseSmoothedScoringReport(results, oppIDStrengthPairs, n) if (results != None) else np.NaN
    
    def _retryGameSimulation(self, cmds, cardinality, timeout):
        for i in range(10):
            self._logger.debug('Retry number ' + str(i))
            time.sleep(2)
            try:
                return self._executeInParallel(cmds, timeout)
            except ValueError as e:
                self._logger.error(e)
                continue
        
        return None
    
    def gameName(self):
        return 'Checkers' 
    
    def _calcBlockMatrices(self,x):
        forZeros = np.zeros((self.__vectorCardinality,), dtype=np.int)
        forOnes = np.ones((self.__vectorCardinality,), dtype=np.int)
        blockMatrix1 = tuple(np.append(forOnes,forZeros))
        blockMatrix2 = tuple(np.append(forZeros,forOnes))
        x1 = np.compress(blockMatrix1,x,axis=1)
        x2 = np.compress(blockMatrix2,x,axis=1)
        return x1,x2
        
    def getDataSet(self):
        
        #For calculating individual blocks of matrices as [ x1 | x2 ]
        #self.x1 , self.x2 = self._calcBlockMatrices(self.input_data_set)
        self.output_data_set = np.reshape(self.output_data_set, (np.shape(self.output_data_set)[0],1))
        return self.input_data_set, self.output_data_set
        
    def updateDataSet(self, player1, player2, output):
        #update_time1 = time.time()
        a=np.append(player1[0] , player2[0])
        for output_value in output:
            output_value = int(output_value)
            if (output_value==1 or output_value==-1):
                self.input_data_set = np.vstack((self.input_data_set,a))#input data set whose type is a list
                self.output_data_set = np.append(self.output_data_set,output_value)#output dataset whose type is a np array
            else:
                self.input_data_set = np.vstack((self.input_data_set,a))#input data set whose type is a list
                self.input_data_set = np.vstack((self.input_data_set,a))#input data set whose type is a list
                plusOne = 1
                minusOne = -1
                self.output_data_set = np.append(self.output_data_set,plusOne)
                self.output_data_set = np.append(self.output_data_set,minusOne)
        
        return None
    def generatePairsofHeuristics(self,numberOfPairs,mode='train'):
        
        opponents1 = self.generateOpponents(numberOfPairs)
        opponents2 = self.generateOpponents(numberOfPairs)
        player1 = np.zeros((1, self.__vectorCardinality))
        player2 = np.zeros((1, self.__vectorCardinality))
        self.test_data_set = np.empty((0,2*self.__vectorCardinality),float)
        self.output_test_data = np.empty((0,1),int)
        
        if mode == 'test':

            for i in xrange(numberOfPairs):
                player1[0, :] = opponents1[i]#Some correction which makes the game comaptible to process a game given multiple heuristics
                player2[0, :] = opponents2[i]
                a = np.append( player1[0] , player2[0] )
                self.test_data_set = np.vstack ( ( self.test_data_set , a ) )
        
            return self.test_data_set
        
        if mode == 'accuracy':
            temp = self._numPlayoutsPerOpponent
            self._numPlayoutsPerOpponent = 1
            for i in xrange(numberOfPairs):
                r_player1 = self.generateOpponents(1)
                r_player2 = self.generateOpponents(1)
                s = self._simulate(r_player1, r_player2) # simulate a game 
                result = self._calcIndividualGains(s) 
                while(int(result[0][0]) == 0):              # We try another game if the result is a draw .
                    r_player1 = self.generateOpponents(1)
                    r_player2 = self.generateOpponents(1)
                    s = self._simulate(r_player1, r_player2) # simulate a game
                    result = self._calcIndividualGains(s)
                a = np.append( r_player1[0] , r_player2[0] )
                self.test_data_set = np.vstack ( ( self.test_data_set , a ) )
                self.output_test_data = np.append(self.output_test_data, int(result[0][0]))
                
            self._numPlayoutsPerOpponent = temp
            self.output_test_data = np.reshape(self.output_test_data, (np.shape(self.output_test_data)[0],1))
            return self.test_data_set, self.output_test_data
        
        if mode == 'train':
            for i in xrange(numberOfPairs):
                player1[0, :] = opponents1[i]#Some correction which makes the game comaptible to process a game given multiple heuristics
                player2[0, :] = opponents2[i]
                print "Game : ",i
                s = self._simulate(player1, player2) # simulate a game 
                result = self._calcIndividualGains(s)                
                self.updateDataSet( player1, player2, result[0] )#updates the data set
            return None
            
        