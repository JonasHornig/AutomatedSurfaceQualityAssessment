import json
import random

import torch

from . import Logging as LOG

class EpisodeLoader():
    def __init__(self, DataSet, DataSetParameters, InputParameter):
        self.Samples                   : list[str] = DataSetParameters["Samples"]
        self.Classes                   : list[str] = DataSetParameters["Classes"]
        self.Images                    : dict      = DataSet
        self.NumberOfClassesPerEpisode : int       = InputParameter.NumberOfClassesPerEpisode

        self.NumberOfSupportSamplesPerClass    : int = InputParameter.NumberOfSupportSamplesPerClass
        self.NumberOfQuerySamplesPerClass      : int = InputParameter.NumberOfQuerySamplesPerClass
        self.NumberOfValidationSamplesPerClass : int = 1

        self.LabelMap    : dict = {}

        self.Initialize()

    def Initialize(self):
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]

    def GenerateLabelMap(self, OutputPath) -> None:
        self.LabelMap = {}
        for Index, Class in enumerate(self.CombinedClasses):
            self.LabelMap[Class] = Index
        self.LookUpTable = {v:k for k,v in self.LabelMap.items()}
        with open(f"{OutputPath}/Labelmap.json", "w") as f:
            json.dump(self.LabelMap, f)

    def GetRandomEpisode(self, LogFile, Controls, Mode : str = "Training") -> dict:
        '''
        Creates a random episode out of the provided data set
        following Algorithm 1 from Snell et al. (2017)
        doi: doi.org/10.48550/arXiv.1703.05175
        '''
        
        # Assumes that the data set is balanced, meaning the same classes and modes for each input sample.

        if not self.Images:
            raise Exception("No data set available. Load data set before creating episodes.")

        # Generate training tensors with the shape [NC, NS, DINOv2 Feature dimension]
        SelectedClasses = random.sample(self.CombinedClasses, self.NumberOfClassesPerEpisode)
        Episode = {}
        SupportTensors    = []
        QueryTensors      = []
        ValidationTensors = []
        for Class in SelectedClasses:
            MaximumAvailableSamples = self.GetEpisodicparameter(Class, Mode)

            AllIndices   = torch.randperm(MaximumAvailableSamples)
            SupportRange = self.NumberOfSupportSamplesPerClass
            QueryRange   = self.NumberOfSupportSamplesPerClass + self.NumberOfQuerySamplesPerClass
            SupportIndices    : list[int] = AllIndices[             : SupportRange].tolist()
            QueryIndices      : list[int] = AllIndices[SupportRange : QueryRange  ].tolist()
            ValidationIndices : list[int] = AllIndices[QueryRange   :             ].tolist()
        
            IndexedSupportTensors = []
            for Index in SupportIndices:
                IndexedSupportTensors.append(self.Images[Class.split("_")[0]][Class.split("_")[1]][Mode]["DinoFeatures"][Index])
            SupportTensors.append(torch.stack(IndexedSupportTensors))
        
            IndexedQueryTensors = []
            for Index in QueryIndices:
                IndexedQueryTensors.append(self.Images[Class.split("_")[0]][Class.split("_")[1]][Mode]["DinoFeatures"][Index])
            QueryTensors.append(torch.stack(IndexedQueryTensors))
        
            IndexedValidationTensors = []
            for Index in ValidationIndices:
                IndexedValidationTensors.append(self.Images[Class.split("_")[0]][Class.split("_")[1]][Mode]["DinoFeatures"][Index])
            ValidationTensors.append(torch.stack(IndexedValidationTensors))
        
        Episode["SupportTensor"]    = torch.stack(SupportTensors)
        Episode["QueryTensor"]      = torch.stack(QueryTensors)
        Episode["ValidationTensor"] = torch.stack(ValidationTensors)

        if Controls.WriteDetailedDebugInfo:
            LOG.TrainingEpisode(LogFile, Controls, SelectedClasses, SupportIndices, QueryIndices, ValidationIndices, Episode)

        return Episode

    def GetEpisodicparameter(self, CombinedClass, Mode):
        MaximumAvailableSamples = len(self.Images[CombinedClass.split("_")[0]][CombinedClass.split("_")[1]][Mode]["Names"])

        # Calculate the continuous solution
        AvailableSamples = int(0.75 * MaximumAvailableSamples)
        SupportToQueryRatio = self.NumberOfSupportSamplesPerClass/self.NumberOfQuerySamplesPerClass

        RequiredNumberOfSamples = self.NumberOfSupportSamplesPerClass + self.NumberOfQuerySamplesPerClass

        if RequiredNumberOfSamples < AvailableSamples:
            return MaximumAvailableSamples

        NQ_Ideal = AvailableSamples / (SupportToQueryRatio + 1)
        NS_Ideal = SupportToQueryRatio * NQ_Ideal

        # Search integer solutions near continuous solution
        BestRatioError = float("inf")
        SearchRadius   = 3
        SolutionFound  = False
        for NS in range(max(1, int(NS_Ideal) - SearchRadius), int(NS_Ideal) + SearchRadius):
            for NQ in range(max(1, int(NQ_Ideal) - SearchRadius), int(NQ_Ideal) + SearchRadius):
                if NS + NQ > AvailableSamples:
                    continue
                RatioError = abs(NS/NQ - SupportToQueryRatio)
                if RatioError < BestRatioError:
                    BestRatioError     = RatioError
                    self.NumberOfSupportSamplesPerClass = NS
                    self.NumberOfQuerySamplesPerClass   = NQ
                    SolutionFound = True
        if not SolutionFound:
            self.NumberOfSupportSamplesPerClass = 1
            self.NumberOfQuerySamplesPerClass   = 1

        #breakpoint()

        return MaximumAvailableSamples