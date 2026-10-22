import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from . import Logging as LOG
from . import NeuralNetwork as NN
from . import InteractivePCA as IPCA

class TestHandler():
    def __init__(self, DataSet, EpisodeLoader, Network):
        self.Samples     : list[str] = DataSet.Samples
        self.Classes     : list[str] = DataSet.Classes
        self.Images      : dict      = DataSet.Images

        self.LabelMap    : dict         = EpisodeLoader.LabelMap
        self.LookUpTable : dict         = EpisodeLoader.LookUpTable
        self.GroupOf     : dict         = EpisodeLoader.GroupOf # {<Sample_Class> : class of the network (depends on ClassGrouping)}
        self.Prototypes  : torch.Tensor = Network.Prototypes

    def ExtractData(self, LogFile, Controls, Mode="Testing"):
        # Generates one flat feature tensor with the shape [Total number of images, DINO feature dimension].
        # The classes are concatenated instead of stacked, so they can contain different numbers of images.
        # Row i of the tensor belongs to entry i of Names, Labels (class of the network), LabelIndices and Members (<Sample_Class> of the image).
        self.PreparedData = {"Names": [], "Labels": [], "LabelIndices": [], "Members": [], "FeatureTensor": torch.Tensor()}
        ClassTensors = []
        for Sample in self.Samples:
            for Class in self.Classes:
                Member = f"{Sample}_{Class}"
                Label  = self.GroupOf[Member]
                Images = self.Images[Sample][Class][Mode]
                ClassTensors.append(Images["DinoFeatures"])
                self.PreparedData["Names"]        += Images["Names"]
                self.PreparedData["Labels"]       += [Label] * len(Images["Names"])
                self.PreparedData["LabelIndices"] += [self.LabelMap[Label]] * len(Images["Names"])
                self.PreparedData["Members"]      += [Member] * len(Images["Names"])

        self.PreparedData["FeatureTensor"] = torch.cat(ClassTensors, dim=0)
        if Controls.WriteDetailedDebugInfo:
            LOG.DataExtraction(LogFile, self.PreparedData)

def NetworkTestMain(LogFile, Controls, ProtoNet, TestHandler):
    LogFile.W("\n******************************\n*   Testing the Proto Net    *\n******************************")
    print("\nTesting the trained network")
    NetworkTestApplication(LogFile, Controls, ProtoNet, TestHandler)
    PrincipalComponentAnalysis(Controls, TestHandler)
    PlotSurfaceStateScores(Controls, TestHandler)

def NetworkTestApplication(LogFile, Controls, Network, TestHandler):
    TestHandler.ExtractData(LogFile, Controls, "Testing")
    Network.Encoder.eval() # BatchNorm uses the running statistics from training instead of the statistics of the test data
    with torch.no_grad():
        TestHandler.PreparedData["Embeddings"] = Network.Encoder(TestHandler.PreparedData["FeatureTensor"])

    PrototypeStates = [SurfaceState(TestHandler.LookUpTable[Index]) for Index in range(len(TestHandler.LookUpTable))]
    Classifications, Probabilities, StateProbabilities = ClassifyEmbeddings(TestHandler.PreparedData["Embeddings"], Network.Prototypes, PrototypeStates)
    TestHandler.PreparedData["Probabilities"]      = Probabilities
    TestHandler.PreparedData["Classifications"]    = Classifications
    TestHandler.PreparedData["TrueStates"]         = [SurfaceState(Member) for Member in TestHandler.PreparedData["Members"]]
    TestHandler.PreparedData["PredictedStates"]    = [PrototypeStates[Index] for Index in Classifications]
    TestHandler.PreparedData["StateProbabilities"] = StateProbabilities
    # Rows: true class, columns: predicted class, both in label map order
    TestHandler.PreparedData["ConfusionMatrix"]    = confusion_matrix(TestHandler.PreparedData["LabelIndices"], TestHandler.PreparedData["Classifications"], labels=list(range(len(TestHandler.LookUpTable))))

    LOG.TestApplicationResults(LogFile, Controls, TestHandler)

def ClassifyEmbeddings(Embeddings, Prototypes, PrototypeStates: list[str]) -> tuple[list[int], list[float], list[float]]:
    # Closest prototype, its softmax probability (the confidence) and the surface state confidence of every embedding: the summed
    # probability of all prototypes with the predicted surface state. With one prototype per material and state this is the
    # probability of the state regardless of the material.
    AllProbabilities               = torch.softmax(-NN.EuclideanDistance(Embeddings, Prototypes), dim=1)
    Probabilities, Classifications = AllProbabilities.max(1)
    SameState = torch.tensor([[PrototypeStates[Index] == State for State in PrototypeStates] for Index in Classifications.tolist()])
    return Classifications.tolist(), Probabilities.tolist(), (AllProbabilities * SameState).sum(1).tolist()

def PlotSurfaceStateScores(Controls, TestHandler):
    # Surface state score of every test tile: distance to the nearest polished prototype minus distance to the nearest scratched one.
    # A tile is scratched from 0 on (its nearest prototype is scratched), so the plot shows how clearly the tiles are separated.
    # SurfaceStateScores.png in the run folder
    States = [SurfaceState(TestHandler.LookUpTable[Index]) for Index in range(len(TestHandler.LookUpTable))]
    if not {"Polished", "Scratched"} <= set(States):
        return
    Scratched = torch.tensor([State == "Scratched" for State in States])
    Distances = NN.EuclideanDistance(TestHandler.PreparedData["Embeddings"], TestHandler.Prototypes)
    Scores    = (Distances[:, ~Scratched].min(1).values - Distances[:, Scratched].min(1).values).numpy()
    Members   = np.array(TestHandler.PreparedData["Members"])
    LOG.PlotValueDistribution({(Sample, Class): Scores[Members == f"{Sample}_{Class}"] for Sample in TestHandler.Samples for Class in TestHandler.Classes}, 0,
                              f"{Controls.OutputPath}/SurfaceStateScores.png", LogScale=True,
                              XLabel="Surface state score: distance to the nearest polished minus to the nearest scratched prototype")

def SurfaceState(ClassName: str) -> str:
    # Class names are <Material>_<SurfaceState> or only <SurfaceState>
    return ClassName.split("_")[-1]

def PrincipalComponentAnalysis(Controls, TestHandler, NumberOfComponents: int = 2):
    # Each PCA is fitted on the test set it shows. The prototypes are projected with the PCA of the embeddings.
    Features = TestHandler.PreparedData["FeatureTensor"].numpy()
    Pca      = PCA(n_components=NumberOfComponents).fit(Features)
    FeatureView = {"Name": "DINO features", "Coordinates": Pca.transform(Features), "Variance": Pca.explained_variance_ratio_, "Prototypes": None}
    LOG.PlotPrincipalComponentAnalysis(Controls, TestHandler, FeatureView["Coordinates"], FeatureView["Variance"],
                                       "PCA_DinoFeatures", Title="Extracted Features")

    Embeddings = TestHandler.PreparedData["Embeddings"].numpy()
    Pca        = PCA(n_components=NumberOfComponents).fit(Embeddings)
    EmbeddingView = {"Name": "Embeddings", "Coordinates": Pca.transform(Embeddings), "Variance": Pca.explained_variance_ratio_,
                     "Prototypes": Pca.transform(TestHandler.Prototypes.numpy())}
    LOG.PlotPrincipalComponentAnalysis(Controls, TestHandler, EmbeddingView["Coordinates"], EmbeddingView["Variance"],
                                       "PCA_Embeddings", EmbeddingView["Prototypes"], Title="Embeddings (Test data & Prototypes)")

    if Controls.InteractivePCA:
        IPCA.WriteInteractivePCA(Controls, TestHandler, [EmbeddingView, FeatureView], Controls.InteractivePCAImages)
