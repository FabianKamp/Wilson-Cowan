import os
import glob
import scipy.io
import numpy as np

class DTIdataset():
    """This class loads the DTI data from the SCZ-Dataset."""
    def __init__(self, Group=None, DataDir=None, normalize=True):
        if Group==None:
            print('Please enter Group Name: HC, SCZ or SCZaff.')
            return

        self.DataDir = DataDir
        if DataDir is None:
            self.DataDir = r'C:\Users\Kamp\Documents\SCAN\Thesis\DTI-Data'
            print('Take default data directory: ', self.DataDir)
        else:
            if os.path.isdir(DataDir):
                self.DataDir=DataDir
            else:
                print('Data Directory does not exist.')

        self.GroupDir = os.path.join(self.DataDir, Group)
        self._loadData(normalize)

    def _loadData(self, normalize):
        """Loads subject data, normalizes and takes the average of connectivity and length matrices.
        Averaged Cmat and LengthMat is saved in self.Cmat and self.LengthMat."""
        self.SubData = {}
        self._loadSubjectFiles()
        self.NumSubjects = len(self.SubData['Cmats'])

        if self.NumSubjects == 0:
            print('No connectivity matrices found.')
            return

        # Normalize connectivitiy matrices
        if normalize:
            self._normalizeCmats()

        # Average connectivity and length matrices
        self.Cmat = self._getAverage(self.SubData['Cmats'])
        self.LengthMat = self._getAverage(self.SubData['LengthMats'])

    def _loadSubjectFiles(self):
        """Function loads all of the subject data into self.SubData dictionary"""

        self.CMFiles = glob.glob(os.path.join(self.GroupDir, '**/*CM.mat'), recursive=True)
        self.LENFiles = glob.glob(os.path.join(self.GroupDir, '**/*LEN.mat'), recursive=True)

        self.SubData['Cmats'] = []
        self.SubData['LengthMats'] = []

        for cm, len in zip(self.CMFiles, self.LENFiles):
            CMFile = scipy.io.loadmat(cm)
            for key in CMFile:
                if isinstance(CMFile[key], np.ndarray):
                    self.SubData['Cmats'].append(CMFile[key])
                    break
            LENFile = scipy.io.loadmat(len)
            for key in LENFile:
                if isinstance(LENFile[key], np.ndarray):
                    self.SubData['LengthMats'].append(LENFile[key])
                    break

    def _normalizeCmats(self, method="max"):
        if method == "max":
            for c in range(self.NumSubjects):
                maximum = np.max(self.SubData['Cmats'][c])
                self.SubData['Cmats'][c]  = self.SubData['Cmats'][c] / maximum

    def _getAverage(self, Mats):
        mat = np.zeros(Mats[0].shape)
        for m in Mats:
            mat += m
        mat = mat/len(Mats)
        return mat

def MEGlowEnv(Subject , cFreq, DataDir):
    if DataDir is None:
        DataDir = r"C:\Users\Kamp\Documents\SCAN\Thesis\MEG-Data\Archive\results_"+ Subject
    LowMagFile = glob.glob(os.path.join(DataDir, f"FrqCarrier-{cFreq}_low-ampl-env.npy"))
    if not LowMagFile:
        LowMagFile = glob.glob(os.path.join(DataDir, f"{cFreq}*.npy"))
    if not LowMagFile:
        raise Exception("File not found.")
    LowMagFile = LowMagFile[0]
    LowEnv = np.load(LowMagFile)
    return LowEnv




