import torch

from sklearn.decomposition import PCA

from . import Logging as LOG
from . import NeuralNetwork as NN

class ValidationHandler():
    def __init__(self, DataSet, DataSetParameters, EpisodeLoader, Network):
        self.Samples     : list[str] = DataSetParameters["Samples"]
        self.Classes     : list[str] = DataSetParameters["Classes"]
        self.Images      : dict      = DataSet.Images

        self.LabelMap    : dict         = EpisodeLoader.LabelMap
        self.LookUpTable : dict         = EpisodeLoader.LookUpTable
        self.Prototypes  : torch.Tensor = Network.Prototypes

        self.DataPrepared : bool = False

        self.PostInit()

    def PostInit(self):
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]

    def ExtractData(self, LogFile, Controls, Mode="Testing"):
        # Generates one flat feature tensor with the shape [Total number of images, DINO feature dimension].
        # The classes are concatenated instead of stacked, so they can contain different numbers of images.
        # Row i of the tensor belongs to entry i of MasterIndices, Names, Labels and LabelIndices.
        self.PreparedData = {"MasterIndices": [], "Names": [], "Labels": [], "LabelIndices": [], "FeatureTensor": torch.empty((1, 1), dtype=torch.float32)}
        RunningIndex = 0
        ClassTensors = []
        for Sample in self.Samples:
            for Class in self.Classes:
                Label      = f"{Sample}_{Class}"
                LabelIndex = self.LabelMap[Label]
                Features = []
                for Index in range(self.Images[Sample][Class][Mode]["DinoFeatures"].shape[0]):
                    ActiveFeatureTensor = self.Images[Sample][Class][Mode]["DinoFeatures"][Index]
                    ActiveName = self.Images[Sample][Class][Mode]["Names"][Index]

                    Features.append(ActiveFeatureTensor)
                    self.PreparedData["MasterIndices"].append(RunningIndex)
                    self.PreparedData["Names"].append(ActiveName)
                    self.PreparedData["Labels"].append(Label)
                    self.PreparedData["LabelIndices"].append(LabelIndex)
                    RunningIndex += 1
                ClassTensors.append(torch.stack(Features))

        self.PreparedData["FeatureTensor"] = torch.cat(ClassTensors, dim=0)
        if Controls.WriteDetailedDebugInfo:
            LOG.DataExtraction(LogFile, self.PreparedData)
        self.DataPrepared = True

def NetworkValidationMain(LogFile, Controls, ProtoNet, ValidationHandler):
    LogFile.W("\n******************************\n*  Validating the Proto Net  *\n******************************")
    print("\nValidating the trained network")
    NetworkTestApplication(LogFile, Controls, ProtoNet, ValidationHandler)
    PrincipalComponentAnalysis(LogFile, Controls,  ValidationHandler)

def NetworkTestApplication(LogFile, Controls, Network, ValidationHandler):
    ValidationHandler.ExtractData(LogFile, Controls, "Testing")
    Network.Encoder.eval() # BatchNorm uses the running statistics from training instead of the statistics of the test data
    with torch.no_grad():
        ValidationHandler.PreparedData["Embeddings"] = Network.Encoder.forward(ValidationHandler.PreparedData["FeatureTensor"])

    DistanceMatrix     = NN.EuclideanDistance(ValidationHandler.PreparedData["Embeddings"], Network.Prototypes)
    ProbabilityMatrix  = torch.nn.functional.softmax(-DistanceMatrix, dim=1)

    # Closest prototype and highest probability per test image, computed once for all images
    MinimumDistances, Classifications = DistanceMatrix.min(1)
    MaximumProbabilities, _           = ProbabilityMatrix.max(1)
    MasterIndices = ValidationHandler.PreparedData["MasterIndices"]
    ValidationHandler.PreparedData["Distances"]       = MinimumDistances[MasterIndices].tolist()
    ValidationHandler.PreparedData["Probabilities"]   = MaximumProbabilities[MasterIndices].tolist()
    ValidationHandler.PreparedData["Classifications"] = Classifications[MasterIndices].tolist()

    LOG.TestApplicationResults(LogFile, ValidationHandler.PreparedData, ValidationHandler.LookUpTable)

def PrincipalComponentAnalysis(LogFile, Controls, ValidationHandler, NumberOfComponents:int = 2):
    Pca = PCA(n_components=NumberOfComponents)
    ReshapedFeatureTensor = ValidationHandler.PreparedData["FeatureTensor"].reshape(-1, Controls.FeatureDimension).detach().cpu().numpy()
    Pca.fit(ReshapedFeatureTensor)

    ValidationHandler.PreparedData["TransformedFeatureTensor"] = Pca.transform(ReshapedFeatureTensor)
    ValidationHandler.PreparedData["ExplainedVarianceRatio"]   = Pca.explained_variance_ratio_
    LOG.PlotPrincipalComponentAnalysis(Controls, ValidationHandler, ValidationHandler.PreparedData["TransformedFeatureTensor"], "PCA_DinoFeatures", Title="PCA of the DINO features (test set)")

    Pca = PCA(n_components=NumberOfComponents)
    ReshapedEmbeddings = ValidationHandler.PreparedData["Embeddings"].detach().cpu().numpy()
    Pca.fit(ReshapedEmbeddings)
    ValidationHandler.PreparedData["TransformedEmbeddings"]  = Pca.transform(ReshapedEmbeddings)
    ValidationHandler.PreparedData["ExplainedVarianceRatio"] = Pca.explained_variance_ratio_
    TransformedPrototypes  = Pca.transform(ValidationHandler.Prototypes.detach().cpu().numpy())
    LOG.PlotPrincipalComponentAnalysis(Controls, ValidationHandler, ValidationHandler.PreparedData["TransformedEmbeddings"], "PCA_Embeddings", TransformedPrototypes, Title="PCA of the embeddings (test set) with class prototypes")