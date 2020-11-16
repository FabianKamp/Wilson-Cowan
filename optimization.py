from deap import creator, base, tools
import random
from multiprocessing import Pool
import numpy as np
from time import time
import time
from collections import OrderedDict
from neurolib.utils.loadData import Dataset
from neurolib.models.wc import WCModel
from SignalAnalysis import Signal
import scipy

class optimization():
    def __init__(self, DataFile, method='FC'):
        """
        Loads Data and set parameters for evolution. 
        Initialiazes first generation. 
        :params Empiric Data File
        """
        assert method in ['FC', 'CCD'], 'Fitting must be FC or CCD'
        self.method = method  
        # Load empirical Data
        self.empMat = np.load(DataFile)
        self.lowpass = 0.2
        
        if method == 'FC':
            # Delete subcortical regions: 
                #Hippocampus: 41 - 44
                #Amygdala: 45-46
                #Basal Ganglia: 75-80
                #Thalamus: 81-82
            # Attention: AAL indices start with 1
            exclude = list(range(40,44)) + list(range(44,46)) + list(range(74,80)) + list(range(80,82))
            self.empMat = np.delete(self.empMat, exclude, axis=0)
            self.empMat = np.delete(self.empMat, exclude, axis=1)

        # Wilson Cowan Parameter Ranges that get optimized
        # Params that get optimized, keys must be equal to wc parameter names
        # NMDA Parameters: exc_ext, c_excinh
        # Gaba Parameters: c_inhinh, c_inhexc
        #self.ParamRanges = OrderedDict({'sigma_ou': [0.001, 0.01], 'exc_ext':[0.5,0.75], 'c_excinh':[13.0,18.0], 'c_inhexc':[13.0,18.0], 'c_inhinh':[1.0,5.0]})          
        self.ParamRanges = OrderedDict({'sigma_ou':[0.001, 0.25], 'K_gl':[0.0, 5.0], 'exc_ext':[0.45,0.9]})

        # Setup genetic algorithm and wc simulator
        self._setup_ga()
        self._setup_wc() 

    def _setup_wc(self):
        """
        Set ups fixed parameters for the wilson cowan model. 
        """
        ds = Dataset("hcp")
        self.Dmat = ds.Dmat
        self.Cmat = ds.Cmat
        # Wilson Cowan Params that stay fixed, keys must be equal to wc parameter names
        # Time in milliseconds
        self.simdt = 1.
        self.simfsample = (1./self.simdt) * 1000
        self.fixedParams = {'duration':2.0*60.0*1000, 'dt':self.simdt}
        self.FreqBand = [8, 12]
        
    def _setup_ga(self):
        """
        Sets up parameters for the genetic algorithm.
        """
        self.toolbox = base.Toolbox()
        self._initializePopulation()
        # genetic algorithm settings
        self.NPopinit = 30
        self.crossPortion = 0.4
        self.mutPortion = 0.4
        self.elitPortion = 0.1
        self.CxPB = .75 # Crossing Over probability
        self.MutPB = .75 # Mutation Probability
        
        # Sigma of gaussian distribution with which attributes are mutated
        self.MutSigma = [(up-low)/4 for low,up in self.ParamRanges.values()]

        # Genetic Operations
        # Register genetic operators with default arguments in toolbox
        self.toolbox.register("model", self._parallel_wc) 
        self.toolbox.register("select", tools.selBest)                  
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=self.MutSigma, indpb=0.5)                      
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)


    def _initializePopulation(self):
        """
        Initializes the population
        """
        # create individuals
        # Using FC, the correlation between FC matrices is maximized
        if self.method == 'FC':
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        elif self.method == 'CCD':
        # Using CCD, the correlation between FC matrices is minimized
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Initialize individuals
        self.toolbox.register("individual", self._initRandParam, creator.Individual, self.ParamRanges)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) # n is defined in main function
        
    def _initRandParam(self, individual, ParamRanges):
        """
        Function to initialize individuals with random parameters
        :param individual: deap individual object
        :param allParams: list of all parameter ranges
        :return: deap individual with random parameter settings within the parameter ranges
        """
        RandParams = []
        for Range in ParamRanges.values():
            rparameter = np.round(random.uniform(Range[0], Range[1]),4)
            RandParams.append(rparameter)
        Individual = individual(RandParams)
        return Individual

    def optimize(self):
        """
        Main Function. Fits/optimizes the wilson cowan model with respect
        to the empirical data
        """
        print('Started Optimization.')
        Generation = 0
        # Create the Population with n individuals
        self.pop = self.toolbox.population(n=self.NPopinit)

        # Compute Wilson Cowan Model for individuals in initial population
        with Pool(processes=20) as p:
            wc_results = p.map(self.toolbox.model, self.pop)
            fitnesses = p.map(self.getFit, wc_results)

        # Assign fitness value to each individual in the pop
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = (fit,)  

        fits = [ind.fitness.values[0] for ind in self.pop]
        print(f"Generation {Generation}, size {len(self.pop)}")
        print('Initial Mean Fit: ', np.mean(fits))
        print('Initial Max Fit: ', np.max(fits), ', Parameters: ', self.pop[np.argmax(fits)])
        print('Initial Pop: ', self.pop)

        while len(self.pop) >= 3:
            Generation += 1            
            # elite offspring 
            elite = self.toolbox.select(self.pop, int(len(self.pop)*self.elitPortion))  
            elite = list(map(self.toolbox.clone, elite))
            
            # Select crossover offspring
            crossover = self.toolbox.select(self.pop, int(len(self.pop)*self.crossPortion))  
            crossover = list(map(self.toolbox.clone, crossover))
            random.shuffle(crossover)

            # Select crossover offspring
            mutants = self.toolbox.select(self.pop, int(len(self.pop)*self.mutPortion))  
            mutants = list(map(self.toolbox.clone, mutants))
            
            # Apply crossover and mutation
            self.applyCrossover(crossover)
            self.applyMutation(mutants)

            # Replace Population with new Individuals
            self.pop = elite + crossover + mutants

            # Find invalid fitness values in offspring
            invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]

            # Reevaluate the fitness of invalid offspring
            with Pool(processes=20) as p:
                wc_results = p.map(self.toolbox.model, invalid_ind)
                fitnesses = p.map(self.getFit, wc_results)

            # Reasign Fitness Value
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
            
            fits = [ind.fitness.values[0] for ind in self.pop]
            print(f"Generation {Generation}, size {len(self.pop)}")
            print('Mean Fit: ', np.mean(fits))
            print('Max Fit: ', np.max(fits), ', Parameters: ', self.pop[np.argmax(fits)])
            print('Population: ', self.pop)
        
    def applyCrossover(self, Individuals):
        """
        Applies CrossOver function to list of individuals that is passed in 
        :return list of individuals 
        """
        for ind1, ind2 in zip(Individuals[::2], Individuals[1::2]):
            if np.random.rand() < self.CxPB:
                (child1, child2) = self.toolbox.mate(ind1, ind2)
                del child1.fitness.values
                del child2.fitness.values

    def applyMutation(self, Individuals):
        """
        Apply Mutation to list of individuals that is passed in
        :return list of individuals
        """
        for mutant in Individuals:
            if np.random.rand() < self.MutPB:
                # indpb is the probability for each attribute to be mutated
                self.toolbox.mutate(mutant)
                mutant[:] = np.round(mutant,4)
                # Reset values that are outside ParamRange
                mutant[:] = [value if low<=value<=up else min(max(value, low), up) 
                            for (low, up), value in zip(self.ParamRanges.values(), mutant[:])]
                del mutant.fitness.values            

    def _parallel_wc(self, params):
        """
        Runs the wilson cowan model. Filters it to the alpha frequency band
        and caclulates the fitting matrix - FC or CCD
        """
        # Set random seed
        random.seed(0)
        # Init Model
        wc = WCModel(Cmat = self.Cmat, Dmat = self.Dmat)
        # set fix parameter 
        for key, parameter in self.fixedParams.items():
            wc.params[key] = parameter
        # set individual parameter
        for key, value in zip(self.ParamRanges.keys(), params):
            if key == 'exc_ext':
                wc.params[key] = np.repeat(value, wc.params['N'])
            else:
                wc.params[key] = value
        # run model
        wc.run(chunkwise=True, append=True)
        # get exc_time courses 
        exc_tc = wc.outputs.exc
        # transform to signal
        signal = Signal(exc_tc, fsample=self.simfsample, lowpass=self.lowpass)
        # Calculate FC or CCD matrix on signal
        mat = getattr(signal, 'get'+self.method)(Limits=self.FreqBand, conn_mode='lowpass-corr')
        return mat

    def getFit(self, simMat):
        """
        Evaluates the fit of each subject in the population
        """
        # Fit FC using correlation Coefficient between empirical and simulated Data
        if self.method == 'FC': 
            # Calculate Correlation between empirical and simulated FC
            simFC = simMat
            empFC = self.empMat
            
            rows = empFC.shape[-1]
            idx = np.triu_indices(rows, k=1)

            empValues = empFC[idx]
            simValues = simFC[idx]

            corr, _ = scipy.stats.pearsonr(empValues, simValues)
            return corr
        
        # Fit CCD using KSD distance
        elif fitting == 'CCD': 
            dist = self._getKSD(self.empMat, simVal)
            return dist
    
    def _getKSD(self, simMat):
        """
        Returns the Kolmogorov-Smirnov distance which ranges from 0-1
        :param simCCD: numpy ndarray containing the simulated CCD
        :return: KS distance between simulated and empirical data
        """
        from scipy.stats import ks_2samp
        if self.empMat.shape != simMat.shape:
            raise ValueError("Input matrices must have the same shape.")

        rows = self.empMat.shape[-1]
        idx = np.triu_indices(rows, k=1)

        empValues = self.empMat[idx]
        simValues = simMat[idx]

        # Bin values
        bins = np.arange(0,1,0.01)
        simIdx = np.digitize(simValues, bins)
        empIdx = np.digitize(empValues, bins)

        binnedSim = bins[simIdx-1]
        binnedEmp = bins[empIdx-1]

        KSD = ks_2samp(binnedEmp, binnedSim)[0] # Omits the p-value and outputs the KSD distance
        return KSD

    
