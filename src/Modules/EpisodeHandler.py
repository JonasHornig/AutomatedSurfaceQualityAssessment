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
        self.ValidationFraction                : float = InputParameter.ValidationFraction
        self.ClassGrouping                     : str   = InputParameter.ClassGrouping

        # Variable initialization
        self.LabelMap          : dict = {}
        self.TrainingIndices   : dict = {} # {<Sample_Class> : tensor of image indices used for support and query samples}
        self.ValidationIndices : dict = {} # {<Sample_Class> : tensor of image indices held back for validation}

    def Initialize(self, LogFile):
        # The images are stored per material and surface state (<Sample_Class>). The network is trained on groups of them:
        #   MaterialAndState - every <Sample_Class> is its own class, e.g. 18201-GJSCastIron_Polished
        #   State            - the materials are pooled, one class per surface state, e.g. Polished
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]
        if self.ClassGrouping == "MaterialAndState":
            self.GroupMembers = {Member: [Member] for Member in self.CombinedClasses}
        elif self.ClassGrouping == "State":
            self.GroupMembers = {Class: [f"{Sample}_{Class}" for Sample in self.Samples] for Class in self.Classes}
        else:
            raise ValueError(f"ClassGrouping has to be MaterialAndState or State, got {self.ClassGrouping}")
        self.Groups  = list(self.GroupMembers)
        self.GroupOf = {Member: Group for Group, Members in self.GroupMembers.items() for Member in Members}

        # With a single class per episode the loss is always zero, as every query is trivially assigned to the only prototype
        if not 2 <= self.NumberOfClassesPerEpisode <= len(self.Groups):
            Message = f"WARNING! - 2 <= N_C (= {self.NumberOfClassesPerEpisode}) <= K (= {len(self.Groups)}) is untrue. Setting N_C = K"
            print(Message)
            LogFile.W(Message)
            self.NumberOfClassesPerEpisode = len(self.Groups)

        self.SplitValidationSet()
        self.BuildPools()
        self.CalculateEpisodeSize(LogFile)

    def BuildPools(self, Mode: str = "Training"):
        # Training and validation features of every class, pooled over its members. The names identify the images in the log.
        self.TrainingPool, self.TrainingPoolNames, self.ValidationPool = {}, {}, {}
        for Group, Members in self.GroupMembers.items():
            TrainingFeatures, Names, ValidationFeatures = [], [], []
            for Member in Members:
                Sample, ClassName = Member.split("_", 1)
                Features = self.Images[Sample][ClassName][Mode]["DinoFeatures"]
                TrainingFeatures.append(Features[self.TrainingIndices[Member]])
                ValidationFeatures.append(Features[self.ValidationIndices[Member]])
                Names += [Index if len(Members) == 1 else f"{Member}/{Index}" for Index in self.TrainingIndices[Member].tolist()]
            self.TrainingPool[Group]      = torch.cat(TrainingFeatures)
            self.TrainingPoolNames[Group] = Names
            self.ValidationPool[Group]    = torch.cat(ValidationFeatures)

    def CalculateEpisodeSize(self, LogFile):
        '''
        Determines the number of support (NS) and query (NQ) samples per class once for the whole run.
        The sizes are the same for all classes, so the class tensors can be stacked. The configured NS and NQ are kept
        unless the smallest class (counting only the images that are not held back for validation) has fewer images.
        '''
        AvailableSamples = min(len(Pool) for Pool in self.TrainingPool.values())
        NS = self.NumberOfSupportSamplesPerClass
        NQ = self.NumberOfQuerySamplesPerClass
        if NS + NQ > AvailableSamples:
            NS, NQ = self.ReduceEpisodeSizes(AvailableSamples)
            Message = (f"WARNING! - The smallest class only has {AvailableSamples} images available for support and query samples. "
                       f"Reducing the episodes from N_S = {self.NumberOfSupportSamplesPerClass}, N_Q = {self.NumberOfQuerySamplesPerClass} to N_S = {NS}, N_Q = {NQ}")
            print(Message)
            LogFile.W(Message)
        self.NumberOfSupportSamplesPerClass = NS
        self.NumberOfQuerySamplesPerClass   = NQ

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

    def GetValidationSet(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns all held back validation images of all classes [Number of validation images, Feature dimension] and their labels from the label map
        Features = [self.ValidationPool[Group] for Group in self.Groups]
        Labels   = [torch.full((len(self.ValidationPool[Group]),), self.LabelMap[Group], dtype=torch.long) for Group in self.Groups]
        return torch.cat(Features), torch.cat(Labels)

    def GetTrainingSet(self) -> list[torch.Tensor]:
        # Training features of every class (without the validation images), ordered by the label map
        return [self.TrainingPool[Group] for Group in sorted(self.Groups, key=lambda Group: self.LabelMap[Group])]

    def GenerateLabelMap(self, OutputPath) -> None:
        self.LabelMap = {}
        for Index, Group in enumerate(self.Groups):
            self.LabelMap[Group] = Index
        self.LookUpTable = {v:k for k,v in self.LabelMap.items()}
        with open(f"{OutputPath}/Labelmap.json", "w") as f:
            json.dump(self.LabelMap, f)

    def GetRandomEpisode(self, LogFile, Controls) -> dict:
        '''
        Creates a random episode with the shape [NC, NS, DINO feature dimension] out of the provided data set
        following Algorithm 1 from Snell et al. (2017)
        doi: doi.org/10.48550/arXiv.1703.05175
        '''

        SelectedClasses = random.sample(self.Groups, self.NumberOfClassesPerEpisode)
        NS, NQ  = self.NumberOfSupportSamplesPerClass, self.NumberOfQuerySamplesPerClass
        Episode = {}
        SupportTensors    = []
        QueryTensors      = []
        ValidationTensors = []
        ValidationLabels  = []
        EpisodeImages     = {} # {<Class> : (support images, query images)}, only used for the log
        for EpisodeLabel, Class in enumerate(SelectedClasses):
            # Support and query samples are only drawn from the images that are not held back for validation
            Pool            = self.TrainingPool[Class]
            ShuffledIndices = torch.randperm(len(Pool))
            SupportIndices  = ShuffledIndices[   : NS   ]
            QueryIndices    = ShuffledIndices[NS : NS+NQ]
            EpisodeImages[Class] = ([self.TrainingPoolNames[Class][Index] for Index in SupportIndices.tolist()],
                                    [self.TrainingPoolNames[Class][Index] for Index in QueryIndices.tolist()])

            SupportTensors.append(   Pool[SupportIndices]         )
            QueryTensors.append(     Pool[QueryIndices]           )
            ValidationTensors.append(self.ValidationPool[Class]   )
            ValidationLabels.append( torch.full((len(self.ValidationPool[Class]),), EpisodeLabel, dtype=torch.long))

        Episode["SupportTensor"]    = torch.stack(SupportTensors)   # Dim: N_C, N_S, Feature dimension
        Episode["QueryTensor"]      = torch.stack(QueryTensors)     # Dim: N_C, N_Q, Feature dimension
        Episode["ValidationTensor"] = torch.cat(ValidationTensors)  # Dim: Number of validation images of the selected classes, Feature dimension
        Episode["ValidationLabels"] = torch.cat(ValidationLabels)   # Dim: Number of validation images of the selected classes. Labels follow the episode's class order

        if Controls.WriteDetailedDebugInfo:
            LOG.TrainingEpisode(LogFile, EpisodeImages, Episode)

        return Episode
