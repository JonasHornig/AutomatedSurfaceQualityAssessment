import json
import random

import torch

from . import Logging as LOG

class EpisodeLoader():
    def __init__(self, DataSet, InputParameter):
        self.Samples                   : list[str] = DataSet.Samples
        self.Classes                   : list[str] = DataSet.Classes
        self.Images                    : dict      = DataSet.Images
        self.NumberOfClassesPerEpisode : int       = InputParameter.NumberOfClassesPerEpisode

        self.NumberOfSupportSamplesPerClass    : int   = InputParameter.NumberOfSupportSamplesPerClass
        self.NumberOfQuerySamplesPerClass      : int   = InputParameter.NumberOfQuerySamplesPerClass
        self.ValidationFraction                : float = 0.25

        # Variable initialization
        self.LabelMap           : dict = {}
        self.TrainingIndices    : dict = {}
        self.TrainingFeatures   : dict = {}
        self.ValidationIndices  : dict = {}
        self.ValidationFeatures : dict = {}

    def Initialize(self, LogFile):
        self.MaximumAvailableTiles = len(self.Images[self.Samples[0]][self.Classes[0]]["Training"]["Names"])
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]
        if not 2 <= self.NumberOfClassesPerEpisode <= len(self.CombinedClasses):
            self.NumberOfClassesPerEpisode = len(self.CombinedClasses)
            print(f"2 <= KC (= {len(self.CombinedClasses)}) <= K (= {self.NumberOfClassesPerEpisode}) untrue.\nSetting KC = K")
            LogFile.W(f"WARNING! 2 <= KC (= {len(self.CombinedClasses)}) <= K (= {self.NumberOfClassesPerEpisode}) untrue - Setting KC = K")
        self.CalculateEpisodeSize("Training", LogFile)
        self.SplitDataSet()

    def CalculateEpisodeSize(self, Mode, LogFile):
        for Sample in self.Samples:
            for Class in self.Classes:
                NumberOfExamples = len(self.Images[Sample][Class][Mode]["Names"])
                if NumberOfExamples < self.MaximumAvailableTiles:
                    self.MaximumAvailableTiles = NumberOfExamples
        AvailableTiles = int(self.MaximumAvailableTiles * (1 - self.ValidationFraction))

        NS = self.NumberOfSupportSamplesPerClass
        NQ = self.NumberOfQuerySamplesPerClass
        if NS + NQ > AvailableTiles:
            NS, NQ = self.ReduceEpisodeSizes(AvailableTiles)
        
        self.NumberOfSupportSamplesPerClass = NS
        self.NumberOfQuerySamplesPerClass   = NQ
        AvailableValidationTiles = self.MaximumAvailableTiles - NS - NQ
        self.NumberOfValidationSamplesPerClass = int(min(AvailableValidationTiles, 200))

    def ReduceEpisodeSizes(self, AvailableSamples: int) -> tuple[int, int]:
        SupportToQueryRatio = self.NumberOfSupportSamplesPerClass / self.NumberOfQuerySamplesPerClass

        NQ_Ideal = AvailableSamples / (SupportToQueryRatio + 1)
        NS_Ideal = SupportToQueryRatio * NQ_Ideal

        BestRatioError = float("inf")
        SearchRadius   = 3
        BestSizes      = (1, 1)
        for NS in range(max(1, int(NS_Ideal) - SearchRadius), int(NS_Ideal) + SearchRadius + 1):
            for NQ in range(max(1, int(NQ_Ideal) - SearchRadius), int(NQ_Ideal) + SearchRadius + 1):
                if NS + NQ > AvailableSamples:
                    continue
                RatioError = abs(NS/NQ - SupportToQueryRatio)
                if RatioError < BestRatioError:
                    BestRatioError = RatioError
                    BestSizes      = (NS, NQ)
        return BestSizes

    def SplitDataSet(self, Mode: str = "Training"):
        '''
        Holds back a fixed, random part of the images of every class as validation set.
        Episodes only draw their support and query samples from the remaining images, so the validation images are never used for training.
        The split is drawn once, so the validation accuracy of all episodes is measured on the same images. It is reproducible through the seed.
        '''
        if not 0 < self.ValidationFraction < 1:
            self.ValidationFraction = 0.25

        for Class in self.CombinedClasses:
            Sample, ClassName = Class.split("_", 1)
            NumberOfImages           = len(self.Images[Sample][ClassName][Mode]["Names"])
            Indices = torch.randperm(NumberOfImages)
            self.TrainingIndices[Class]    = Indices[self.NumberOfValidationSamplesPerClass:]
            self.TrainingFeatures[Class]   = [self.Images[Sample][ClassName][Mode]["DinoFeatures"][Index] for Index in self.TrainingIndices[Class]]
            self.ValidationIndices[Class]  = Indices[:self.NumberOfValidationSamplesPerClass]
            self.ValidationFeatures[Class] = [self.Images[Sample][ClassName][Mode]["DinoFeatures"][Index] for Index in self.ValidationIndices[Class]]

    def GenerateLabelMap(self, OutputPath) -> None: # usage?
        self.LabelMap = {}
        for Index, Class in enumerate(self.CombinedClasses):
            self.LabelMap[Class] = Index
        self.LookUpTable = {v:k for k,v in self.LabelMap.items()}
        with open(f"{OutputPath}/Labelmap.json", "w") as f:
            json.dump(self.LabelMap, f)

    def GetRandomEpisode(self, LogFile, Controls, Mode : str = "Training") -> dict: # usage?
        '''
        Creates a random episode out of the provided data set
        following Algorithm 1 from Snell et al. (2017)
        doi: doi.org/10.48550/arXiv.1703.05175
        '''

        # Generate training tensors with the shape [NC, NS, DINO feature dimension]
        SelectedClasses = random.sample(self.CombinedClasses, self.NumberOfClassesPerEpisode)
        Episode = {}
        SupportTensors    = []
        QueryTensors      = []
        ValidationTensors = []
        ValidationLabels  = []
        EpisodeIndices    = {} # {<Sample_Class> : (support indices, query indices)}, only used for the log
        for EpisodeLabel, Class in enumerate(SelectedClasses):
            Sample, ClassName = Class.split("_", 1)

            # Support and query samples are only drawn from the images that are not held back for validation
            ShuffledIndices = torch.randperm(len(self.TrainingFeatures[Class]))
            SupportIndices    : list[int] = ShuffledIndices[:self.NumberOfSupportSamplesPerClass].tolist()
            QueryIndices      : list[int] = ShuffledIndices[self.NumberOfSupportSamplesPerClass:].tolist()
            ValidationIndices : list[int] = self.ValidationIndices[Class].tolist()
            EpisodeIndices[Class] = (SupportIndices, QueryIndices)

            SupportTensors.append(torch.stack([self.TrainingFeatures[Class][Index] for Index in SupportIndices]))
            QueryTensors.append(torch.stack([self.TrainingFeatures[Class][Index] for Index in QueryIndices]))
            ValidationTensors.append(torch.stack(self.ValidationFeatures[Class]))
            ValidationLabels.append( torch.full((len(ValidationIndices),), EpisodeLabel, dtype=torch.long))

        Episode["SupportTensor"]    = torch.stack(SupportTensors)   # Dim: N_C, N_S, Feature dimension
        Episode["QueryTensor"]      = torch.stack(QueryTensors)     # Dim: N_C, N_Q, Feature dimension
        Episode["ValidationTensor"] = torch.cat(ValidationTensors)  # Dim: Number of validation images of the selected classes, Feature dimension
        Episode["ValidationLabels"] = torch.cat(ValidationLabels)   # Dim: Number of validation images of the selected classes. Labels follow the episode's class order

        if Controls.WriteDetailedDebugInfo:
            LOG.TrainingEpisode(LogFile, EpisodeIndices, Episode)

        return Episode