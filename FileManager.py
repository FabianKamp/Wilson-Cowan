import mat73
import Z_config as config
import os, glob, sys
import numpy as np
import pandas as pd
from utils.SignalAnalysis import Signal
import matplotlib.pyplot as plt

class FileManager():
    """
    Class to manage all file dependencies of this project.
    """
    def __init__(self):
        # Configure according to config File
        self.ParentDir = config.ParentDir
        self.DataDir = config.DataDir
        self.InfoFile = config.InfoFile
        self.NetDir = config.NetDir

        # Get Group IDs from Info sheet
        self.ControlIDs = self.getGroupIDs('CON')
        self.FEPIDs = self.getGroupIDs('FEP')
        self.GroupIDs = {'Control': self.ControlIDs, 'FEP': self.FEPIDs}

        # AAL name file 
        self.RegionNames, self.RegionCodes = self.getRegionNames()
        self.RegionCoordinates = self.getRegionCoords()

        # Load Attributes from config file 
        self.FrequencyBands = config.FrequencyBands
        self.DownFreq = config.DownFreq
        self.GraphMeasures = config.GraphMeasures
        self.SubjectList = config.SubjectList
        self.Frequencies = config.Frequencies
        self.net_version = config.net_version

    def createFileName(self, suffix, filetype, **kwargs):
        """
        Function to create FileName string. The file name and location is inferred from the suffix.
        Creates directories if not existing.
        :param suffix: name suffix of file
        :return: FilePath string
        """     
        # config.mode contains orth-lowpass, orth, etc. Is automatically added to suffix.
        if config.conn_mode and ('no_conn', True) not in list(kwargs.items()):
            suffix += '_' + config.conn_mode
                
        FileName = ''
        for key, val in kwargs.items():
            if key != 'no_conn':
                FileName += key + '-' + str(val) + '_'
        
        FileName += suffix + filetype
        return FileName

    def createFilePath(self, *args):
        Directory = ''
        for arg in args[:-1]:
            Directory = os.path.join(Directory, arg)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)

        FilePath = os.path.join(Directory, args[-1])
        return FilePath

    def exists(self, suffix, filetype, **kwargs):
        FileName = self.createFileName(suffix, filetype, **kwargs)
        if glob.glob(os.path.join(self.ParentDir, f'**/{FileName}'), recursive=True):
            return True
        else:
            return False
    
    def find(self, suffix, filetype, **kwargs):
        FileName = self.createFileName(suffix, filetype, **kwargs)
        InnerPath = glob.glob(os.path.join(self.ParentDir, f'**/{FileName}'), recursive=True)
        if len(InnerPath)>1:
            raise Exception(f'Multiple Files found: {InnerPath}')
        if len(InnerPath)<1:
            raise Exception(f'No File found: {FileName}')
        
        TotalPath = os.path.join(self.ParentDir, InnerPath[0])
        return TotalPath

    def getGroupIDs(self, Group):
        """
        Gets the IDs of the group. Handles the Info-File.
        This function only works with the Info-File supplied in the Info-Folder
        :param Group: str, Group that should be loaded
        :return: list of IDs
        """
        ExcelContent = pd.read_excel(self.InfoFile)
        pos = self._getLocation(ExcelContent, Group)
        IDs = ExcelContent.loc[pos[0] + 2:, pos[1]]
        IDs = list(IDs.dropna())
        return IDs

    def getGroup(self, SubjectNum):
        """
        Returns the Group that Subject ID belongs to
        :param SubjectNum:
        :return:
        """
        if SubjectNum in self.FEPIDs:
            Group = 'FEP'
        elif SubjectNum in self.ControlIDs:
            Group = 'Control'
        else:
            Group = None
            print(f'{SubjectNum} not found in {config.InfoFileName}')
        return Group
    
    def getRegionNames(self):
        AAL2File = config.AAL2NamesFile
        with open(AAL2File, 'r') as file:
            f=file.readlines()
        assert len(f) == 94, 'AAL Name File must contain 94 lines.'        
        labels=[line[:-1].split()[1] for line in f]
        codes =[line[:-1].split()[2] for line in f]
        codes = list(map(int,codes))
        return labels, codes
    
    def getRegionCoords(self): 
        import json
        with open(config.AAL2CoordsFile, 'r') as file: 
            CoordDict = json.load(file)
        return CoordDict

    def _getLocation(self, df, value, all=False):
        """
        Gets the first location (idx) of the value in the dataframe.
        :param df: panda.DataFrame
        :param value: str, searched string
        :param all: if true, returns all occurences of value
        :return: list of indices
        """
        poslist = []
        temp = df.isin([value]).any()
        columnlist = list(temp[temp].index)
        for column in columnlist:
            rows = df[column][df[column] == value].index
            for row in rows:
                poslist.append([row, column])
                if not all:
                    return poslist[0]
        return poslist
     
class MEGManager(FileManager):
    def __init__(self):
        super().__init__()
        # Create the Directory Paths
        self.FcDir = os.path.join(self.ParentDir, 'FunctCon')
        self.GroupStatsFC = os.path.join(self.ParentDir, 'GroupStatsFunctCon')
        self.MSTDir = os.path.join(self.ParentDir, 'MinimalSpanningTree')
        self.BinFcDir = os.path.join(self.ParentDir, 'BinFunctCon')
        self.SplitFcDir = os.path.join(self.ParentDir, 'SplitFunctCon')
        self.MetaDir = os.path.join(self.ParentDir, 'Metastability')
        self.CCDDir = os.path.join(self.ParentDir, 'CCD')
        self.SubjectAnalysisDir = os.path.join(self.ParentDir, 'GraphMeasures', 'SubjectAnalysis')
        self.NetMeasuresDir = os.path.join(self.ParentDir, 'GraphMeasures')
        self.PlotDir = os.path.join(self.ParentDir, 'Plots')
        if len(self.SubjectList) == 0:
            self.SubjectList = self.getSubjectList()

    def getSubjectList(self):
        """Gets the subject numbers of the MEG - Datafiles.
        """
        MEGFiles = glob.glob(os.path.join(self.DataDir, '*AAL94_norm.mat'))
        FileList = [Path.split('/')[-1] for Path in MEGFiles]
        SubjectList = [File.split('_')[0] for File in FileList]
        return SubjectList

    def getFCList(self):
        """Gets the subject numbers of the MEG-FC - Datafiles
        """
        FCFiles = glob.glob(os.path.join(self.FcDir, '*'))
        FCList = [Path.split('/')[-1] for Path in FCFiles]
        return FCList
    
    def loadMatFile(self, Subject):
        """
        Loads the MEG - Signal of specified Subject from Mat file
        :param Subject: Subject ID
        :return: Dictionary containing Signal and Sampling Frequency
        """
        SubjectFile = os.path.join(self.DataDir, Subject + '_AAL94_norm.mat')
        DataFile = mat73.loadmat(SubjectFile)
        fsample = int(DataFile['AAL94_norm']['fsample'])
        signal = DataFile['AAL94_norm']['trial'][0] # Signal has to be transposed
        return signal.T, fsample

class EvolutionManager(FileManager):
    """This class loads the DTI data from the SCZ-Dataset."""
    def __init__(self, Group=None):
        super().__init__()
        if Group==None:
            raise Exception('Please enter Group Name: HC, SCZ or SCZaff.')
        else:
            self.Group = Group
        self.DTIDir = config.DTIDir
        self.MEGDir = config.DataDir

        if not os.path.isdir(self.DTIDir) or not os.path.isdir(self.MEGDir):
            raise Exception('Data Directory does not exist.')

    def loadDTIDataset(self, normalize=True):
        """Loads subject data, normalizes and takes the average of connectivity and length matrices.
        Averaged Cmat and LengthMat is saved in self.Cmat and self.LengthMat."""
        self.SubData = {}
        self._loadDTIFiles()
        self.NumSubjects = len(self.SubData['Cmats'])

        if self.NumSubjects == 0:
            raise Exception('No connectivity matrices found.')

        # Normalize connectivitiy matrices
        if normalize:
            self._normalizeCmats()

        # Average connectivity and length matrices
        self.Cmat = self._getSCAverage(self.SubData['Cmats'])
        self.LengthMat = self._getSCAverage(self.SubData['LengthMats'])

    def _loadDTIFiles(self):
        """Function loads all of the subject data into self.SubData dictionary"""
        import scipy.io

        self.CMFiles = glob.glob(os.path.join(self.DTIDir + self.Group, '**/*CM.mat'), recursive=True)
        self.LENFiles = glob.glob(os.path.join(self.DTIDir + self.Group, '**/*LEN.mat'), recursive=True)

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

    def _getSCAverage(self, Mats):
        mat = np.zeros(Mats[0].shape)
        for m in Mats:
            mat += m
        mat = mat/len(Mats)
        return mat