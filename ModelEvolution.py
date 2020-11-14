import sys 
sys.path.append('/mnt/raid/data/SFB1315/SCZ/rsMEG/code/ScZ')
from utils.SignalAnalysis import Signal
import matplotlib.pyplot as plt
import numpy as np
from neurolib.models.wc import WCModel

def fit_WC(CCD, params):
    """
    Wrapper function to fit the wilson cowan model
    """


class evWC():
    """
    Class to handle the model simulations
    """
    def __init__(self, params, Settings):
        # List of Parameters that will be modified during evolution
        self.evolKeys = Settings['evolKeys']

        # Initialize Wilson-Cowan Model
        self.simFreq = 500 # simulation frequency in seconds
        self.WC = WCModel(Cmat=Settings['Cmat'], Dmat=Settings['Dmat']) # Initialize Wilson Cowan model
        self.NumberStrRegions = self.WC.params['N']

        self.WC.params['duration'] = Settings['Duration'] * 1000 # Duration in ms
        self.WC.params['dt'] = 1000 * (1/self.simFreq) # timestep in ms

        # Runt WC simulations with parameter
        self._runWC(params)

        # Calculate Low-pass Envelope of Carrier Frequency
        self.Limits = Settings['CarrierFreq']
        ExcSignal = Signal(self.WC.outputs.exc, self.simFreq)
        self.ExcEnv = ExcSignal.getLowPassEnvelope(Limits=self.Limits)

        # Downsample Envelope
        self.downNum = 300 # has to be the same as for MEG CCD
        self.dExcEnv = Envelope(self.ExcEnv).downsampleSignal(self.downNum)

        # Get Fit
        empCCD = Settings['empCCD']
        self.Fit = self.getFit(empCCD)

    def _runWC(self, params):
        """
        :param params: list of parameters that are passed into the WC-model
        :return: Envelopes of the excitatory and inhibitory spiking series
        """
        print("Calculating Wilson-Cowan simulation.")
        for idx, key in enumerate(self.evolKeys):
            self.WC.params[key] = params[idx]

        self.WC.params['exc_init'] = 0.05 * np.random.uniform(0, 1, (self.NumberStrRegions, 1)) # setting random initial values
        self.WC.params['inh_init'] = 0.05 * np.random.uniform(0, 1, (self.NumberStrRegions, 1))

        self.WC.run(chunkwise=True, append=True)

    def getFit(self, empData, method='CCD'):
        """Calculate the fit between empirical and WC-model with params as parameter.
        The fit is defined as the KS Distance between the Coherence Connectivity Dynamics of empirical and
        simulated data.
        :param params: list of parameter that are passed into the WC-model
        :return: Fit
        """
        from utils.SignalAnalysis import Envelope
        if method == 'CCD':
            self.simCCD = Envelope(self.dExcEnv).getCCD()
            self.empCCD = empData
            fit = self._getKSD(self.empCCD, self.simCCD)
            print("Fit: ", fit)
            return fit

        if method == 'FC':
            self.simFC = Envelope(self.ExcEnv).getFC()
            self.empFC = empData
            fit = self._getKSD(self.empFC, self.simFC)
            print("Fit: ", fit)
            return fit

    def _getKSD(self, empMat, simMat):
        """
        Returns the Kolmogorov-Smirnov distance which ranges from 0-1
        :param simCCD: numpy ndarray containing the simulated CCD
        :return: KS distance between simulated and empirical data
        """
        from scipy.stats import ks_2samp
        if empMat.shape != simMat.shape:
            raise ValueError("Input matrices must have the same shape.")

        rows = empMat.shape[-1]
        idx = np.triu_indices(rows, k=1)

        empValues = empMat[idx]
        simValues = simMat[idx]

        # Bin values
        bins = np.arange(0,1,0.01)
        simIdx = np.digitize(simValues, bins)
        empIdx = np.digitize(empValues, bins)

        binnedSim = bins[simIdx-1]
        binnedEmp = bins[empIdx-1]

        KSD = ks_2samp(binnedEmp, binnedSim)[0] # Omits the p-value and outputs the KSD distance
        return KSD