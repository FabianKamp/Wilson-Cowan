from deap import creator, base, tools
import random
from multiprocessing import Pool
from ModelEvolution import evWC
import numpy as np
import LoadData
from time import time
from itertools import repeat
import matplotlib.pyplot as plt

##########################
# General Configurations #
##########################

set random seed
random.seed(4)

# Set the CarrierFreq and the Subject number to fit the Model to
CarrierFreq = 20 # TODO iterate over all carrier freqs
Subject = 'S126' # TODO take mean CCD

# genetic algorithm settings
NPop = 1
NGen = 5
CxPB = .65 # Crossing Over probability
MutPB = 1 # Mutation Probability

# list possible mutation strategies
MutStrats = [tools.mutGaussian, tools.mutUniformInt]
# list selection strategies
SelStrats = [tools.selTournament, tools.selRoulette, tools.selBest]
# list crossover stretegies
CxStrats =[tools.cxOnePoint, tools.cxTwoPoint, tools.cxUniform]

# set initial parameter ranges for WC model
K_gl = [2, 5]
signalV = [9,10]
sigma_ou = [0.5, 1.]

RangeParams = [K_gl, signalV, sigma_ou] # TODO Convert to dict
ParamKeys = ["K_gl", "signalV", "sigma_ou"]

def initRandParam(individual, RangeParams):
    """
    Function to initialize individuals with random parameters
    :param individual: deap individual object
    :param allParams: list of all parameter ranges
    :return: deap individual with random parameter settings within the parameter ranges
    """
    RandParams = []
    for RangeParam in RangeParams:
        rparameter = np.round(random.uniform(RangeParam[0], RangeParam[1]),2)
        RandParams.append(rparameter)
    Individual = individual(RandParams)
    return Individual

# create individuals
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox() # Deap syntax: Container where all the elements and operations are "registered"

# Initialize individuals
toolbox.register("individual", initRandParam, creator.Individual, RangeParams)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # n is defined in main function

# Genetic Operations
# Register genetic operators with default arguments in toolbox
toolbox.register("model", evWC)                  # Evaluation function
toolbox.register("mutate", MutStrats[0])            # Choose Mutation function
toolbox.register("select", SelStrats[2])            # Choose Selection function
toolbox.register("mate", CxStrats[2])               # Choose Mating function

#Register statistic functions for logbook
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

log = tools.Logbook()

# Creating the Population
def main():
    # Load DTI Matrices for healthy and scz (functional connectivity is not loaded because we use CCd to fit the Model)
    empDTI = LoadData.DTIdataset(Group="SCZ") # TODO load neurolib dataset
    Dmat = empDTI.LengthMat
    Cmat = empDTI.Cmat

    # Get the empirical MEG - CCD matrix
    fsample = 400
    MEGlowEnv = LoadData.MEGlowEnv(Subject, CarrierFreq, DataDir=None)

    Duration = MEGlowEnv.shape[-1]*(1/fsample) # Duration in seconds
    downNum = 300
    dLowMag = Envelope(MEGlowEnv).downsampleSignal(downNum)
    empCCD = Envelope(dLowMag).getCCD()

    # Configure Settings for the Wilson Cowan Model
    Settings = {'evolKeys': ParamKeys, 'Duration': Duration, 'CarrierFreq': CarrierFreq,
                'Dmat': Dmat, 'Cmat': Cmat, 'empCCD': empCCD}

    pop = toolbox.population(n=NPop)     # n was not defined during the initialization but now

    # Compute Wilson Cowan Model for each individual
    with Pool(processes=2) as pool:
        Models = pool.starmap(toolbox.model, zip(pop, repeat(Settings, NPop)))

    # Assign fitness Values
    fitnesses = [Model.Fit for Model in Models]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)  # assign fitness value to each individual in the pop

    # Performing the Evolution
    fits = [ind.fitness.values[0] for ind in pop]  # list all fitness values
    print(fits)

    # Define Size of Elite list, crossover and mutation portion etc.
    EliteSize = int(len(pop) * 1)#0.1)
    CrossSize = int(len(pop) * 1)# 0.4)
    RestSize = int(len(pop) * 1) #0.6)
    MutSize = int(len(pop) * 1) # 0.3)

    g = 0  # Counter for the number of generations
    while g < 3:
        # Each iteration is a new generation
        g += 1
        print(f"Generation {g}")

        # Compute Generation statistics
        recording = stats.compile(pop)
        log.record(**{'gen':g, 'fits': recording})
        print(log.chapters['fits'].select('avg'))

        # Select offspring
        Elite = toolbox.select(pop, EliteSize)  # select all ind from population
        Elite = list(map(toolbox.clone, Elite))  # clones offsprings

        Crossover = toolbox.select(pop, CrossSize) # select individuals for crossover
        Crossover = list(map(toolbox.clone, Crossover)) # clone offspring
        random.shuffle(Crossover)

        Mutants = toolbox.select(pop, MutSize) # select individuals to mutate
        Mutants = list(map(toolbox.clone, Mutants))

        Rest = tools.selWorst(pop, RestSize) # select the resting individuals
        Rest = list(map(toolbox.clone, Rest))
        random.shuffle(Rest)

        # Select random individuals from rest
        RestN = int(NPop*1) #0.1)
        crossRest = Rest[:RestN]
        mutRest = Rest[RestN:2*RestN]

        # Apply crossover
        applyCrossover(Crossover)
        applyCrossover(crossRest)

        # Apply mutation
        applyMutation(Mutants)
        applyMutation(mutRest)

        # Replace Population with new Individuals
        pop = Elite + Crossover + Mutants + crossRest + mutRest

        # Find invalid fitness values in offspring
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]

        # Reevaluate the fitness of offspring
        with Pool(processes=2) as pool:
            Models = pool.starmap(toolbox.evaluate, zip(invalid_ind, repeat(Settings, len(invalid_ind))))
            fitnesses = [Model.Fit for Model in Models]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # Replace the old population by the offspring
        fits = [ind.fitness.values[0] for ind in pop]
        print(fits)
    return pop

def applyCrossover(Individuals):
    firstChild = []
    secondChild = []
    for child1, child2 in zip(Individuals[::2], Individuals[1::2]):
        if random.random() < CxPB:
            toolbox.mate(child1, child2, CxPB)
            firstChild.append(child1)
            secondChild.append(child2)
            del child1.fitness.values
            del child2.fitness.values

def applyMutation(Individuals):
    mutants = []
    sigma = [0.1,0.1,0.01]
    for mutant in Individuals:
        if random.random() < MutPB:
            toolbox.mutate(mutant, 0, sigma=sigma, indpb=MutPB)
            mutant[:] = np.round(mutant,2)
            mutant[2] = min(1., mutant[2])   # Set maximal noise Value to 1
            mutants.append(mutant)
            del mutant.fitness.values

main()
