import torch

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from . import Logging as LOG
from . import NeuralNetwork as NN

class TestHandler():
    def __init__(self, DataSet, EpisodeLoader, Network):
        self.Samples     : list[str] = DataSet.Samples
        self.Classes     : list[str] = DataSet.Classes
        self.Images      : dict      = DataSet.Images

        self.LabelMap    : dict         = EpisodeLoader.LabelMap
        self.LookUpTable : dict         = EpisodeLoader.LookUpTable
        self.Prototypes  : torch.Tensor = Network.Prototypes

    def ExtractData(self, LogFile, Controls, Mode="Testing"): # usage?
        # Generates one flat feature tensor with the shape [Total number of images, DINO feature dimension].
        # The classes are concatenated instead of stacked, so they can contain different numbers of images.
        # Row i of the tensor belongs to entry i of Names, Labels and LabelIndices.
        self.PreparedData = {"Names": [], "Labels": [], "LabelIndices": [], "FeatureTensor": torch.Tensor()}
        ClassTensors = []
        for Sample in self.Samples:
            for Class in self.Classes:
                Label  = f"{Sample}_{Class}"
                Images = self.Images[Sample][Class][Mode]
                ClassTensors.append(Images["DinoFeatures"])
                self.PreparedData["Names"]        += Images["Names"]
                self.PreparedData["Labels"]       += [Label] * len(Images["Names"])
                self.PreparedData["LabelIndices"] += [self.LabelMap[Label]] * len(Images["Names"])

        self.PreparedData["FeatureTensor"] = torch.cat(ClassTensors, dim=0)
        if Controls.WriteDetailedDebugInfo:
            LOG.DataExtraction(LogFile, self.PreparedData)

def NetworkTestMain(LogFile, Controls, ProtoNet, TestHandler):
    LogFile.W("\n******************************\n*   Testing the Proto Net    *\n******************************")
    print("\nTesting the trained network")
    NetworkTestApplication(LogFile, Controls, ProtoNet, TestHandler)
    PrincipalComponentAnalysis(Controls, TestHandler)

def NetworkTestApplication(LogFile, Controls, Network, TestHandler):
    TestHandler.ExtractData(LogFile, Controls, "Testing")
    Network.Encoder.eval() # BatchNorm uses the running statistics from training instead of the statistics of the test data
    with torch.no_grad():
        TestHandler.PreparedData["Embeddings"] = Network.Encoder(TestHandler.PreparedData["FeatureTensor"])

    # Closest prototype and its softmax probability (the confidence) per test image
    DistanceMatrix = NN.EuclideanDistance(TestHandler.PreparedData["Embeddings"], Network.Prototypes)
    Probabilities, Classifications = torch.softmax(-DistanceMatrix, dim=1).max(1)
    TestHandler.PreparedData["Probabilities"]   = Probabilities.tolist()
    TestHandler.PreparedData["Classifications"] = Classifications.tolist()
    TestHandler.PreparedData["ConfusionMatrix"] = confusion_matrix(TestHandler.PreparedData["LabelIndices"], TestHandler.PreparedData["Classifications"], labels=list(range(len(TestHandler.LookUpTable))))

    LOG.TestApplicationResults(LogFile, Controls, TestHandler)

def PrincipalComponentAnalysis(Controls, TestHandler, NumberOfComponents: int = 2):
    # Each PCA is fitted on the test set it shows. The prototypes are projected with the PCA of the embeddings.
    Features = TestHandler.PreparedData["FeatureTensor"].numpy()
    Pca      = PCA(n_components=NumberOfComponents).fit(Features)
    LOG.PlotPrincipalComponentAnalysis(Controls, TestHandler, Pca.transform(Features), Pca.explained_variance_ratio_, "PCA_DinoFeatures", Title="Extracted Features")

    Embeddings = TestHandler.PreparedData["Embeddings"].numpy()
    Pca        = PCA(n_components=NumberOfComponents).fit(Embeddings)
    LOG.PlotPrincipalComponentAnalysis(Controls, TestHandler, Pca.transform(Embeddings), Pca.explained_variance_ratio_, "PCA_Embeddings", Pca.transform(TestHandler.Prototypes.numpy()), Title="Embeddings (Test data & Prototypes)")
