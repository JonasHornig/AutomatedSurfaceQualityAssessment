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
        self.ValidationFraction                : float = InputParameter.ValidationFraction
        self.ReportedReductions : set = set() # Episode size reductions that were already written to the log

        self.LabelMap          : dict = {}
        self.TrainingIndices   : dict = {} # {<Sample_Class> : tensor of image indices used for support and query samples}
        self.ValidationIndices : dict = {} # {<Sample_Class> : tensor of image indices held back for validation}

        self.Initialize()

    def Initialize(self):
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]
        # With a single class per episode the loss is always zero, as every query is trivially assigned to the only prototype
        if not 2 <= self.NumberOfClassesPerEpisode <= len(self.CombinedClasses):
            raise ValueError(f"NumberOfClassesPerEpisode has to be between 2 and the number of classes in the data set ({len(self.CombinedClasses)}), got {self.NumberOfClassesPerEpisode}")
        self.SplitValidationSet()

    def SplitValidationSet(self, Mode: str = "Training"):
        '''
        Holds back a fixed, random part of the images of every class as validation set.
        Episodes only draw their support and query samples from the remaining images, so the validation images are never used for training.
        The split is drawn once, so the validation accuracy of all episodes is measured on the same images. It is reproducible through the seed.
        '''
        if not 0 < self.ValidationFraction < 1:
            raise ValueError(f"ValidationFraction has to be between 0 and 1, got {self.ValidationFraction}")
        for Class in self.CombinedClasses:
            Sample, ClassName = Class.split("_", 1)
            NumberOfImages           = len(self.Images[Sample][ClassName][Mode]["Names"])
            NumberOfValidationImages = max(1, round(self.ValidationFraction * NumberOfImages))
            if NumberOfImages - NumberOfValidationImages < 2:
                raise ValueError(f"Class {Class} only contains {NumberOfImages} {Mode} images. After holding back {NumberOfValidationImages} for validation, at least 2 have to remain for one support and one query sample.")
            Indices = torch.randperm(NumberOfImages)
            self.ValidationIndices[Class] = Indices[:NumberOfValidationImages]
            self.TrainingIndices[Class]   = Indices[NumberOfValidationImages:]

    def GetValidationSet(self, Mode: str = "Training") -> tuple[torch.Tensor, torch.Tensor]:
        # Returns all held back validation images of all classes [Number of validation images, Feature dimension] and their labels from the label map
        Features = []
        Labels   = []
        for Class in self.CombinedClasses:
            Sample, ClassName = Class.split("_", 1)
            Features.append(self.Images[Sample][ClassName][Mode]["DinoFeatures"][self.ValidationIndices[Class]])
            Labels.append(torch.full((len(self.ValidationIndices[Class]),), self.LabelMap[Class], dtype=torch.long))
        return torch.cat(Features), torch.cat(Labels)

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

        # Generate training tensors with the shape [NC, NS, DINO feature dimension]
        SelectedClasses = random.sample(self.CombinedClasses, self.NumberOfClassesPerEpisode)
        NS, NQ          = self.GetEpisodeSizes(SelectedClasses, Mode, LogFile)
        Episode = {}
        SupportTensors    = []
        QueryTensors      = []
        ValidationTensors = []
        ValidationLabels  = []
        EpisodeIndices    = {} # {<Sample_Class> : (support indices, query indices)}, only used for the log
        for EpisodeLabel, Class in enumerate(SelectedClasses):
            Sample, ClassName = Class.split("_", 1)
            Features          = self.Images[Sample][ClassName][Mode]["DinoFeatures"]

            # Support and query samples are only drawn from the images that are not held back for validation
            ShuffledIndices = self.TrainingIndices[Class][torch.randperm(len(self.TrainingIndices[Class]))]
            SupportIndices    : list[int] = ShuffledIndices[   : NS   ].tolist()
            QueryIndices      : list[int] = ShuffledIndices[NS : NS+NQ].tolist()
            ValidationIndices : list[int] = self.ValidationIndices[Class].tolist()
            EpisodeIndices[Class] = (SupportIndices, QueryIndices)

            SupportTensors.append(   Features[SupportIndices]   )
            QueryTensors.append(     Features[QueryIndices]     )
            ValidationTensors.append(Features[ValidationIndices])
            ValidationLabels.append( torch.full((len(ValidationIndices),), EpisodeLabel, dtype=torch.long))

        Episode["SupportTensor"]    = torch.stack(SupportTensors)   # Dim: N_C, N_S, Feature dimension
        Episode["QueryTensor"]      = torch.stack(QueryTensors)     # Dim: N_C, N_Q, Feature dimension
        Episode["ValidationTensor"] = torch.cat(ValidationTensors)  # Dim: Number of validation images of the selected classes, Feature dimension
        Episode["ValidationLabels"] = torch.cat(ValidationLabels)   # Dim: Number of validation images of the selected classes. Labels follow the episode's class order

        if Controls.WriteDetailedDebugInfo:
            LOG.TrainingEpisode(LogFile, EpisodeIndices, Episode)

        return Episode

    def GetEpisodeSizes(self, SelectedClasses, Mode, LogFile) -> tuple[int, int]:
        '''
        Determines the number of support (NS) and query (NQ) samples per class for one episode.
        The sizes are the same for all selected classes, so the class tensors can be stacked, and they are based on the smallest selected class
        (counting only the images that are not held back for validation).
        The configured NS and NQ are used whenever the smallest class allows it. They are never overwritten, so a small class only affects the episodes it is part of.
        '''
        AvailableSamples = min(len(self.TrainingIndices[Class]) for Class in SelectedClasses)

        NS = self.NumberOfSupportSamplesPerClass
        NQ = self.NumberOfQuerySamplesPerClass
        if NS + NQ > AvailableSamples:
            NS, NQ = self.ReduceEpisodeSizes(AvailableSamples)
            if (NS, NQ, AvailableSamples) not in self.ReportedReductions:
                self.ReportedReductions.add((NS, NQ, AvailableSamples))
                Message = (f"WARNING! - The smallest class of an episode only has {AvailableSamples} {Mode} images available for support and query samples. "
                           f"Reducing the episode from N_S = {self.NumberOfSupportSamplesPerClass}, N_Q = {self.NumberOfQuerySamplesPerClass} to N_S = {NS}, N_Q = {NQ}")
                print(Message)
                LogFile.W(Message)
        return NS, NQ

    def ReduceEpisodeSizes(self, AvailableSamples: int) -> tuple[int, int]:
        # Finds the integer NS and NQ with NS + NQ <= AvailableSamples that best keep the configured support to query ratio
        SupportToQueryRatio = self.NumberOfSupportSamplesPerClass / self.NumberOfQuerySamplesPerClass

        # Calculate the continuous solution
        NQ_Ideal = AvailableSamples / (SupportToQueryRatio + 1)
        NS_Ideal = SupportToQueryRatio * NQ_Ideal

        # Search integer solutions near continuous solution
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