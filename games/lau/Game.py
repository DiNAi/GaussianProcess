import numpy as np
import matplotlib.pyplot as plt
import time
import pyGPs
from pyGPs.Validation import valid
from Simulator import GameSimulator as GS, OthelloSimulator as OS,  CheckersSimulator as CS
import scipy.optimize as opt
from scipy.stats import norm
from random import randint
SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
DATACOLOR = [0.12109375, 0.46875, 1., 1.0]

class Sim_Game(OS.OthelloSimulator):
    def __init__(self):
        
        self.mmtreeDepth = 3
        self.numPlaysPerOpponent = 3
        self.number_of_opponents = 2
        self.const_c = np.sqrt((np.pi * np.log(2.0))/2.0)

        self.wins = np.empty((0,1),float)
        self.actual_wins = np.empty((0,1),float)
        self.loss = []
        self.difference = []
        self.varaiances = []
        self.accuracy = []
        self.runs = []
        self.means = []
        
    def Run_Game(self):
            
        print ("Choice 0 : Othello Game with Vector Cardinality 5 .")
        print ("Choice 1 : Othello Game with Vector Cardinality 10 .")
        print ("Choice 2 : Checkers Game with Vector Cardinality 11 .")
        
        choice = int(raw_input("Please Enter Your Choice : "))
        if choice == 1:
            self.Vector_cardinality = 10
            self.gameSimulator = OS.OthelloSimulator(['OthelloTest.exe'], self.Vector_cardinality, self.mmtreeDepth, self.numPlaysPerOpponent, 35, 20, 0.0065, 2000)
            gs = GS.GameSimulator(['OthelloTest.exe'], self.Vector_cardinality, self.mmtreeDepth, self.numPlaysPerOpponent)
            
        elif choice == 2 :
            
            self.Vector_cardinality = 11
            self.gameSimulator = CS.CheckersSimulator(['CheckersTest.exe'],self.Vector_cardinality, self.mmtreeDepth, self.numPlaysPerOpponent, 35, 20, 0.0065, 2000)
            gs = GS.GameSimulator(['CheckersTest.exe'], self.Vector_cardinality, self.mmtreeDepth, self.numPlaysPerOpponent)
            
        else :
            
            self.Vector_cardinality = 5
            self.gameSimulator = OS.OthelloSimulator(['OthelloTest.exe'], self.Vector_cardinality, self.mmtreeDepth, self.numPlaysPerOpponent, 35, 20, 0.0065, 2000)
            gs = GS.GameSimulator(['OthelloTest.exe'], self.Vector_cardinality, self.mmtreeDepth, self.numPlaysPerOpponent)

        self.opponents = gs.generateOpponents(self.number_of_opponents)
        self.benchmark_opponents = gs.generateOpponents(50)
        for heuristic in self.opponents:
            player = np.zeros((1, self.Vector_cardinality))
            player[0, :] = heuristic
            self.wins = np.append(self.wins,self.gameSimulator.benchMarkHeuristic(player,self.benchmark_opponents))
        
        flipped_wins = self.wins[::-1]                              #Flip the array so that we can get the max index from last.
        self.top_heuristic = self.opponents[-np.argmax(flipped_wins)-1]
    
            
        
        self.gameSimulator.playTournament(self.opponents)                 #simulates a tournament for the given set of heuristics 
        
     
        self.x , self.y = self.gameSimulator.getDataSet()#Fetches the training data set
        self.model = pyGPs.gp.GPC()
        
        fMean = pyGPs.mean.Zero()
        gMean = pyGPs.mean.MeanPref(fMean)                      #Prior Mean is set to 0
            
        fCov = pyGPs.cov.RBF(log_ell=2.0, log_sigma=0.25)       #Prior CoVairiance is calculating using RBF
        gCov = pyGPs.cov.CovPref(fCov)
        
        self.model.setData(self.x, self.y)
        
        self.model.setPrior(mean=gMean, kernel=gCov)            #Now we set the prior mean and cov functions as the new Preference mean and Cov function

        #Calculate the posterior of the model (i.e.) combining the prior and likelihood(default : Erf) and then calculates the inference (default : Expectation Propagation)
        #Returns the log Marginal likelihood, its derivatives and also the posterior structure
        self.model.getPosterior()
        self.model.optimize()
        
        test_heuristic , actual_classes = self.gameSimulator.generatePairsofHeuristics(1000,mode='accuracy')
        ym,ys,fm,f2,llp = self.model.predict(xs = test_heuristic)
        predicted_classes = np.sign(ym)
        accu = valid.ACC(predicted_classes,actual_classes)                                              #For calculating the accuracy of prediction
        print accu*100          
        
          
        self.new_heuristic_set = np.empty((0, self.Vector_cardinality))     #Initialsing the list of new heuristics found
        
        
        for i in xrange(100):                                    #for 100 Runs
            t1 = time.time()
            print 'No. of Runs :',i
            
            
            new_heuristic, mean , variance = self.get_new_heuristic(gs.generateOpponents(1))                       #Compute a new heuristic which Maximizes BALD Obj.
            
            self.means.append(mean)
            self.varaiances.append(variance)
            
            
            player = np.zeros((1, self.Vector_cardinality))
            player[0, :] = new_heuristic
            for times in xrange(5):
                opp =  np.zeros((1, self.Vector_cardinality))
                opp[0, :] = self.opponents[randint(0,self.opponents.shape[0]-1)]
                self.gameSimulator.playPairings(player, opp)                 #Play New Heuristic Vs Old heuristics
            
            
            
            
            self.x , self.y = self.gameSimulator.getDataSet()    #Fetches the updated training data set
            
            self.model = pyGPs.gp.GPC()
            self.model.setData(self.x, self.y)
            self.model.setPrior(mean=gMean, kernel=gCov)
            self.model.getPosterior()                                           #learn the model
            self.model.optimize()

            
            self.new_heuristic_set = np.append(self.new_heuristic_set, player, axis = 0)    #Storing the newly found heuristics so far.
            self.opponents = np.append(self.opponents, player, axis = 0)
            
            self.wins = np.append(self.wins,self.gameSimulator.benchMarkHeuristic(player,self.benchmark_opponents))
            flipped_wins = self.wins[::-1]
            self.top_heuristic = self.opponents[-np.argmax(flipped_wins)-1]
            self.actual_wins = np.append(self.actual_wins,np.amax(self.wins))
            ymu, ys2, fmu, fs2, lp = self.model.predict(xs = test_heuristic)
            predicted_classes = np.sign(ymu)                       
            acc = valid.ACC(predicted_classes,actual_classes)
            print np.sum(predicted_classes),np.sum(actual_classes)
            print "The Accuracy of predictions is ",(acc*100),"%"                           #For calculating the accuracy of prediction
            
            self.accuracy.append(acc*100)
            self.runs.append(i)
            self.plotResults(self.runs,self.means,"Mean Vs No. of Runs","Mean","No. of Runs","Mean.pdf")
            self.plotResults(self.runs,self.varaiances,"Varaince Vs No. of Runs","Variance","No. of Runs","Variance.pdf")
            self.plotResults(self.runs,self.accuracy,"Accuracy Vs No. of Runs","Accuracy","No. of Runs","Accuracy.pdf")
            self.plotResults(self.runs,self.wins,"Winners Score Vs No. of Runs","Score","No. of Runs","Wins.pdf")
            self.plotResults(self.runs,self.actual_wins,"Winners Score Vs No. of Runs","Score","No. of Runs","Actual Wins.pdf")
            self.plot_error_region(np.array(self.means), np.array(self.varaiances),self.runs)
            
            print 'Time for this run was : ', round(time.time()-t1,3)
        
        
        print self.wins
        print self.actual_wins
        print self.loss
        print self.means
        print self.varaiances
        print self.accuracy
        
    
        
    def BALD_Objective(self,heuristic):
        if heuristic.ndim == 1:
            heuristic = np.reshape(heuristic, (1,heuristic.shape[0])) #reshaping since the package changes shape as column vector.
        heuristic = self.merge_heuristic(self.top_heuristic,heuristic)
        fmu, fs2 ,mean , variance, lp = self.model.predict(xs = heuristic)
    
        var = (variance + 1.0) ** (-0.5)
        
        err_func = norm.cdf( mean * var )         #calculate the cumulative standard normal distribution
        term1 =  (((-1.0) *err_func * np.log2(err_func)) - ((1.0 - err_func) * np.log2(1.0 - err_func)))  #Binary Entropy Function and this is the first term of BALD
        
        temp = variance + np.square(self.const_c)   
        temp1 = self.const_c * ( temp ** (-0.5))            #For the second term in BALD
        temp2 = (2.0*temp) ** (-1.0)                      
        term2 = np.exp(-1.0 * np.square(mean) * temp2) * temp1  

        objective = (term1 - term2) * (-1.0)                #Using -1 since we need to maximize this objective
      
        return objective[0][0]
    
    

    
     
    def get_new_heuristic(self, heuristic):
        
        optimal_heuristic = opt.fmin_bfgs(self.BALD_Objective, x0 = heuristic , disp = False)               #Maximize the BALD Objective
        heuristic = np.reshape(optimal_heuristic, (1,optimal_heuristic.shape[0]))                           #reshaping since the package changes shape as column vector.
        optimal_heuristic = self.merge_heuristic(self.top_heuristic,heuristic)
        mean, variance, fmu, fs2, lp = self.model.predict(xs = optimal_heuristic)
        return heuristic,float(mean[0][0]),float(variance[0][0])    
                  
            
    def merge_heuristic(self,x1,x2):
        a = np.append(x1,x2)
        self.merged_heu = np.empty((0,2*self.Vector_cardinality),float)
        self.merged_heu = np.vstack((self.merged_heu,a))
        return self.merged_heu
    
    
    def _calcBlockMatrices(self,x):
        
        forZeros = np.zeros((self.Vector_cardinality,), dtype=np.int)
        forOnes = np.ones((self.Vector_cardinality,), dtype=np.int)
        blockMatrix1 = tuple(np.append(forOnes,forZeros))
        blockMatrix2 = tuple(np.append(forZeros,forOnes))
        x1 = np.compress(blockMatrix1,x,axis=1)
        x2 = np.compress(blockMatrix2,x,axis=1)
        return x1,x2
    
    
    def plotResults(self,x,y,title,yaxis,xaxis,filename):
        fig = plt.figure()
        fig.suptitle(yaxis, fontsize=14, fontweight='bold')

        if yaxis == 'Score':
            x = np.arange(len(y))
            x = list(x * 3)
            
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title(title)
        plt.xlim(xmin=0)
        plt.xlim(xmax=max(x))
        plt.ylim(ymin=min(y))
        if filename == 'Accuracy.pdf':
            plt.ylim(ymax=100)
        else:
            plt.ylim(ymax=max(y)*1.3)
    
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.plot(x, y)
        plt.savefig(filename)
        return None
    
    def plot_error_region(self,mean,variance,runs):
        fig = plt.figure()
        fig.suptitle('For Heuristic Benchmarking', fontsize=14, fontweight='bold')
        std_dev = variance ** 0.5
        
            
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        y2 = mean + (std_dev*2.0)
        y1 =  mean - (std_dev*2.0)
        plt.xlim(xmin=0)
        plt.xlim(xmax=max(runs))
        plt.ylim(ymin=y1.min()-0.3)
        plt.ylim(ymax=y2.max()+0.3)
        ax.set_title('Error region')
        ax.set_xlabel('rounds')
        ax.set_ylabel('mean with 2XS.D')
        ax.plot(runs,mean,color=MEANCOLOR)
        ax.plot(runs, y1, runs, y2, color='black')
        ax.fill_between(runs,y1,y2 ,where = y2>=y1,facecolor=SHADEDCOLOR, interpolate=True)
        plt.savefig('error.pdf')
        return None
        
   
if __name__ == "__main__":    
    
    t1=time.time()
    obj = Sim_Game()
    obj.Run_Game()
    t2=time.time()
    print "time of execution :",t2-t1,"Sec(s)"