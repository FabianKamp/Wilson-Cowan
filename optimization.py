from deap import creator, base, tools
from multiprocessing import Pool
import numpy as np
from datetime import datetime
import time, pickle, random, sys
sys.path.append('/mnt/raid/data/SFB1315/SCZ/rsMEG/code/Signal')
from SignalAnalysis import Signal
from collections import OrderedDict
from neurolib.utils.loadData import Dataset
from neurolib.models.wc import WCModel
import scipy

class optimization():
    def __init__(self, DataFile, mode='FC', method='corr'):
        """
        Loads Data and set parameters for evolution. 
        Initialiazes first generation. 
        :params Datafile - Empiric Data File
        :params mode, 'FC' or 'CCD'
        :params method, 'corr' or 'ksd'
        """
        print('Setting up optimization.')
        assert (mode in ['FC', 'CCD'] and method in ['corr', 'ksd']), 'Mode must be FC or CCD  and method corr or ksd.'
        assert not ((mode=='CCD') & (method == 'corr')), 'For CCD data you can only use the ksd method.'
        self.method = method 
        self.mode = mode 
        
        random.seed(0)
        
        # Load empirical Data
        print('Data File ', DataFile.split('/')[-1])
        self.empData = np.load(DataFile)
        self.lowpass = 0.2
        
        # exclude Subcortical regions
        if mode == 'FC':
            # Subcortical regions: Hippocampus: 41 - 44, Amygdala: 45-46, Basal Ganglia: 75-80, Thalamus: 81-82
            # Attention: AAL indices start with 1
            exclude = list(range(40,46)) + list(range(74,82))
            self.empData = np.delete(self.empData, exclude, axis=0)
            self.empData = np.delete(self.empData, exclude, axis=1)

        # Wilson Cowan Parameter Ranges that get optimized, keys must be equal to wc parameter names
        # NMDA Parameters: exc_ext, c_excinh
        # Gaba Parameters: c_inhinh, c_inhexc
        # self.ParamRanges = OrderedDict({'sigma_ou':[0.001, 0.25], 'K_gl':[0.0, 5.0], 'exc_ext':[0.45,0.9]})
        #self.ParamRanges = OrderedDict({'sigma_ou':[0.001, 0.25], 'K_gl':[0.0, 5.0], 'f_gaba':[0.,1.], 'f_nmda':[0.,1.]})
        self.ParamRanges = OrderedDict({'K_gl':[0.0, 5.0], 'f_nmda':[0.,1.], 'f_gaba':[0.,1.]})
        print('Fitting Parameters and Parameter Ranges ', self.ParamRanges)

        # Output Files
        self.logfile = "logs/" + mode + '.' + datetime.now().strftime("%d.%m.%Y.%H.%M.%S") + ".pkl"
        self.outputfile = "results/" + mode + '.' + datetime.now().strftime("%d.%m.%Y.%H.%M.%S") + ".npy"

        # Setup genetic algorithm and wc simulator
        self._setup_ga()
        self._setup_wc()         
        self.logbook.record(fittingparams=self.ParamRanges)
        self.processes = 25

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
        self.fixedParams = {'duration':3.0*60*1000, 'dt':self.simdt, 'exc_ext':0.5, 'sigma_ou':0.01}
        #self.fixedParams = {'duration':3.0*60*1000, 'dt':self.simdt}
        self.FreqBand = [8, 12]
        self.logbook.record(freq=self.FreqBand, defaultparams=WCModel().params, fixedparams=self.fixedParams)
        
    def _setup_ga(self):
        """
        Sets up parameters for the genetic algorithm.
        """
        self.toolbox = base.Toolbox()
        self._initializePopulation()
        # genetic algorithm settings
        self.NPopinit = 100
        self.NPop = 50
        self.NGen = 50 
        self.crossPortion = 0.4
        self.mutPortion = 0.4
        self.elitPortion = 0.1

        # rank selection parameter s
        self.cx_s = 1.5
        self.cx_u = 3
        self.mut_s = 1.5
        self.mut_u = 3
        
        # Genetic Operations
        # Register genetic operators with default arguments in toolbox
        self.toolbox.register("selBest", tools.selBest) 
        self.toolbox.register("selRank", self.selRank) 
        
        self.toolbox.register("model", self._parallel_wc) 
        self.toolbox.register("evaluate", self.getFit)                
        
        self.toolbox.register("mutate", self.mutate)                      
        self.toolbox.register("mate", self.mate)

        # Logbook 
        self.logbook = tools.Logbook()

    def _initializePopulation(self):
        """
        Initializes the population
        """
        # create individuals
        # Using FC, the correlation between FC matrices is maximized
        if self.method == 'corr':
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        # Using CCD, the correlation between FC matrices is minimized
        elif self.method == 'ksd':
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
    
    def _bestfit(self, fitnesses):
        """
        Helper function that returns best fit with respect to the used method. 
        ksd -> min, corr -> max
        """
        if self.method == 'corr':
            return np.max(fitnesses)
        elif self.method == 'ksd': 
            return np.min(fitnesses)

    def _argbestfit(self,fitnesses): 
        """
        Helper function that returns index of the best fit with respect to the used method. 
        ksd -> min, corr -> max
        """
        if self.method == 'corr':
            return np.argmax(fitnesses)
        elif self.method == 'ksd': 
            return np.argmin(fitnesses)
    
    def _logStats(self, generation): 
        """
        Save generation stats to logbook
        """        
        fits =  [ind.fitness.values[0] for ind in self.pop]
        best_ind = [ind for ind in self.toolbox.selBest(self.pop, 50)]
        record = {'avg': np.mean(fits), 'sd': np.std(fits),
                'max': {'fits': np.max(fits), 'params': self.pop[np.argmax(fits)][:]},
                'min': {'fits': np.min(fits), 'params': self.pop[np.argmin(fits)][:]},
                'best individuals': {'fits':[ind.fitness.values[0] for ind in best_ind], 'params':[ind[:] for ind in best_ind]}}
        self.logbook.record(gen=generation, individuals=len(self.pop), **record)

    def optimize(self):
        """
        Main Function. Fits/optimizes the wilson cowan model with respect
        to the empirical data
        """
        print('Started Optimization.\nInitial Generation')
        start = time.time()
        # Create the Population with n individuals
        self.pop = self.toolbox.population(n=self.NPopinit)
        # Compute Wilson Cowan Model for individuals in initial population
        with Pool(processes=20) as p:
            wc_results = p.map(self.toolbox.model, self.pop)
            fitnesses = p.map(self.toolbox.evaluate, wc_results)
        
        # Save highest fit array
        bestidx = self._argbestfit([fitnesses])
        np.save(self.outputfile, wc_results[bestidx])
        
        # Assign fitness value to each individual in the pop
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = (fit,)         
        self._logStats(generation = 0)

        for generation in range(1, self.NGen):  
            print('Generation: ', generation)
            # Select best NPop individuals
            self.pop = self.toolbox.selBest(self.pop, self.NPop)
            self._logStats(generation=generation)
            
            # Print best Fits
            bestfit = self._bestfit([ind.fitness.values[0] for ind in self.pop])
            print(f"Mean Fit: {self.logbook.select('avg')[-1]}")
            print(f"Best Fit: {bestfit}")
            print(f"Best Ind: {self.toolbox.selBest(self.pop, 1)[0]}")

            # Select parents
            parents = self.toolbox.selRank(self.pop, s=self.cx_s, u=self.cx_u)  
            parents = list(map(self.toolbox.clone, parents))
            random.shuffle(parents)
            # Select mutants 
            mutants = self.toolbox.selRank(self.pop, s=self.mut_s, u=self.mut_u)  
            mutants = list(map(self.toolbox.clone, mutants))
            
            # Apply mate and mutation
            self.toolbox.mate(parents)
            self.toolbox.mutate(mutants)
            offspring = mutants + parents 
            
            # Find invalid fitness values in offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            # Reevaluate the fitness of invalid offspring
            with Pool(processes=self.processes) as p:
                wc_results = p.map(self.toolbox.model, invalid_ind)
                fitnesses = p.map(self.toolbox.evaluate, wc_results)
            
            # Save Array with highest fit
            if self._bestfit(fitnesses+[bestfit]) != bestfit:
                bestidx = self._argbestfit(fitnesses)
                np.save(self.outputfile, wc_results[bestidx])

            # Reasign Fitness Value
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

            # Merge old and new population
            self.pop = offspring + self.pop

        # Evaluate Last Generation and Print final best fit
        print('Last Generation Completed.')
        self.pop = self.toolbox.selBest(self.pop, self.NPop)
        self._logStats(generation=generation+1) 
        bestfit = self._bestfit([ind.fitness.values[0] for ind in self.pop])
        print(f"Mean Fit: {self.logbook.select('avg')[-1]}")
        print(f"Best Fit: {bestfit}")   
        print(f"Best Ind: {self.toolbox.selBest(self.pop, 1)[0]}")

        # Save logbook to file
        with open(self.logfile, 'wb') as file: 
            pickle.dump(self.logbook, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Optimization Done.\nTime: {time.time()-start}')       
        
    def selRank(self, population, s, u):
        """
        Rank based Selection as described in 'Introduction to Evolutionary Computing' p.82
        :params s parameter of the rank selection - suggested 1.5
        :params u parameter of the rank selection - number of offspring of fittest individual
        """
        # Rank fits
        fits = [ind.fitness.values[0] for ind in population]
        idx = np.argsort(fits)
        ranks = np.zeros(len(fits))
        ranks[idx] = np.arange(len(fits))

        # Calculate Selection Probability and select individuals
        selProbs = [(2-s)/u + (2*rank*(s - 1))/(u*(u-1)) for rank in ranks]
        selected = [ind for prob, ind in zip(selProbs, population) if np.random.rand()<=prob]        
        return selected
        
    def mate(self, Individuals):
        """
        Applies CrossOver function to list of individuals 
        """
        for child1, child2 in zip(Individuals[::2], Individuals[1::2]):
            tools.cxBlend(child1, child2, alpha=0.5)
            # Reset values that are outside ParamRange
            for child in [child1, child2]: 
                child[:] = [value if low<=value<=up else low + np.random.rand()*(up-low)
                            for (low, up), value in zip(self.ParamRanges.values(), child[:])]
                child[:] = np.round(child,4)            
            del child1.fitness.values
            del child2.fitness.values

    def mutate(self, Individuals):
        """
        Apply Mutation to list of individuals
        """
        for mutant in Individuals:
            # Sigma of gaussian distribution with which attributes are mutated
            # Sigma is choosen following the range of the parameter and its fitting value (high fits -> small sigma)
            # Minimal weighting value is 0.3
            mutSigma = [up-low for low,up in self.ParamRanges.values()]
            if mutant.fitness.weights[0]==1.0:
                weight = np.array(1-mutant.fitness.values[0])
                weight[weight<0.3] = 0.3
                mutSigma = list(weight*np.array(mutSigma))
            else: 
                weight = np.array(mutant.fitness.values[0])
                weight[weight<0.3] = 0.3
                mutSigma = list(weight*np.array(mutSigma))
            # indpb is the probability for each attribute to be mutated                
            tools.mutGaussian(mutant, mu=0, sigma=mutSigma, indpb=0.5)
            mutant[:] = np.round(mutant,4)
            # Reset values that are outside ParamRange
            mutant[:] = [value if low<=value<=up else low + np.random.rand()*(up-low)
                        for (low, up), value in zip(self.ParamRanges.values(), mutant[:])]
            del mutant.fitness.values

    def f_gaba(self, param): 
        """
        Function to translate f_gaba factor to WC-parameter space
        """
        c_ie_range = [8, 16]
        c_ii_range = [0, 6]
        c_ie = np.min(c_ie_range) + float(np.diff(c_ie_range)) * param
        c_ii = np.min(c_ii_range) + float(np.diff(c_ii_range)) * param
        return c_ie, c_ii
    
    def f_nmda(self, param):
        """
        Function to translate f_nmda factor to WC-parameter space
        """
        #p_e_range = [0.45, 0.9]
        #p_i_range = [0, 0.01]
        c_ei_range = [5, 20]
        #p_e = np.min(p_e_range) + float(np.diff(p_e_range)) * param
        #p_i = np.min(p_i_range) + float(np.diff(p_i_range)) * param
        c_ei = np.min(c_ei_range) + float(np.diff(c_ei_range)) * param
        #return p_e, c_ei           
        return c_ei

    def _parallel_wc(self, params):
        """
        Runs the wilson cowan model. Filters it to the alpha frequency band
        and caclulates the fitting matrix - FC or CCD
        """
        # Set random seed
        #random.seed(0)
        # Init Model
        wc = WCModel(Cmat = self.Cmat, Dmat = self.Dmat)
        # set fix parameter 
        for key, value in self.fixedParams.items():
            wc.params[key] = value
            if key == 'exc_ext':
                wc.params[key] = np.repeat(value, wc.params['N'])
        # set individual parameter
        for key, value in zip(self.ParamRanges.keys(), params):
            if key == 'exc_ext':
                wc.params[key] = np.repeat(value, wc.params['N'])
            # Gaba and NMDA factors
            elif key == 'f_gaba':
                wc.params['c_inhexc'], wc.params['c_inhinh'] = self.f_gaba(value)
            elif key == 'f_nmda':
                wc.params['c_excinh'] = self.f_nmda(value)
                #p_e, wc.params['c_excinh'] = self.f_nmda(value)
                #wc.params['exc_ext'] = np.repeat(p_e, wc.params['N'])
                #wc.params['inh_ext'] = np.repeat(p_i, wc.params['N'])                
            else:
                wc.params[key] = value
        # run model
        wc.run(chunkwise=True, append=True)
        # get exc_time courses 
        exc_tc = wc.outputs.exc
        # transform to signal
        signal = Signal(exc_tc, fsample=self.simfsample, lowpass=self.lowpass)
        # Calculate FC or CCD matrix on signal
        if self.mode == 'FC': 
            mat = signal.getEnvFC(Limits=self.FreqBand, conn_mode='lowpass-corr')
        elif self.mode == 'CCD': 
            mat = signal.getCCD(Limits=self.FreqBand)
        
        return mat

    def getFit(self, simMat):
        """
        Evaluates the fit of each subject in the population
        """
        # Fit FC using correlation Coefficient between empirical and simulated Data
        if self.method == 'corr': 
            # Calculate Correlation between empirical and simulated FC
            simFC = simMat
            empFC = self.empData            
            rows = empFC.shape[-1]
            idx = np.triu_indices(rows, k=1)
            empValues = empFC[idx]
            simValues = simFC[idx]
            corr, _ = scipy.stats.pearsonr(empValues, simValues)
            return corr
        
        # Fit CCD using KSD distance
        elif self.method == 'ksd': 
            dist = self._getKSD(simMat)
            return dist
    
    def _getKSD(self, simMat):
        """
        Returns the Kolmogorov-Smirnov distance which ranges from 0-1
        :param simCCD: numpy ndarray containing the simulated CCD
        :return: KS distance between simulated and empirical data
        """
        from scipy.stats import ks_2samp
        # Get Histogram of empirical and simulated Data
        simhist = self._get_hist_distr(simMat)
        if self.mode == 'FC':
            emphist = self._get_hist_distr(self.empData)
        elif self.mode == 'CCD': 
            emphist = self.empData        
        distance, _ = ks_2samp(emphist, simhist) # Omits the p-value and outputs the KSD distance
        return distance
    
    def _get_hist_distr(self, data):
        """
        Calculates the histogram distribution of data. 
        This function is used for KSD computation.
        :params data, symmetric matrix as nd.array
        :returns array with histogram values
        """
        # Get histogramm 
        hist_idx = np.triu_indices(data.shape[0], k=1)
        hist, bin_edges = np.histogram(data[hist_idx], bins=np.arange(0,1,0.001))
        # midpoint of bin_edges 
        bins = bin_edges[:-1] + np.diff(bin_edges)/2
        hist_distr = np.repeat(bins, hist.astype('int'))
        return hist_distr     
