import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

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

    def ExtractData(self, LogFile, Controls, Mode="Testing"):
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

        for Img in self.Images["18201"]["Polished"]["Training"]["Names"]:
            if "2310602045" in Img:
                print(f"Training: {Img}")

        for Img in self.Images["18201"]["Polished"]["Testing"]["Names"]:
            if "2310602045" in Img:
                print(f"Testing: {Img}")

def NetworkTestMain(LogFile, Controls, ProtoNet, TestHandler):
    LogFile.W("\n******************************\n*   Testing the Proto Net    *\n******************************")
    print("\nTesting the trained network")
    NetworkTestApplication(LogFile, Controls, ProtoNet, TestHandler)
    PrincipalComponentAnalysis(Controls, TestHandler)
    OcclusionSensitivity(LogFile, Controls, ProtoNet, TestHandler)

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

def OcclusionSensitivity(LogFile, Controls, ProtoNet, TestHandler, Index: int = 0, Mode: str = "Testing", WindowSize: int = 64, Stride: int = 16):
    Label      = TestHandler.PreparedData["Labels"][Index]
    Sample     = Label.split("_")[0]
    Class      = Label.split("_")[1]
    Name       = TestHandler.PreparedData["Names"][Index]
    ImagePath  = f"{Controls.DataSetPath}/Images/{Sample}/{Class}/{Mode}/{Name}"

    print(Name)

    PilImage   = Image.open(ImagePath).convert("RGB")
    ImageArray = np.array(PilImage)
    ImageHeight, ImageWidth = ImageArray.shape[:2]

    def GetConfidence(PilImg: Image.Image) -> float:
        DinoPreprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        InputTensor = DinoPreprocessor(PilImg).unsqueeze(0)   # [1, 3, H, W]
        with torch.no_grad():
            DinoFeatures = ProtoNet.DinoModel(InputTensor)    # [1, 768]
            Embedding    = ProtoNet.Encoder(DinoFeatures)     # [1, EmbeddingDim]
            Distances    = NN.EuclideanDistance(Embedding, ProtoNet.Prototypes)  # [1, NC]
            Probabilities = F.softmax(-Distances, dim=1)      # [1, NC]

        PredictedIndex = TestHandler.LabelMap[Label]
        return Probabilities[0, PredictedIndex].item()

    BaselineConfidence = GetConfidence(PilImage)

    HeatMap = np.zeros((ImageHeight, ImageWidth), dtype=np.float32)
    Counts  = np.zeros((ImageHeight, ImageWidth), dtype=np.float32)

    YPositions = range(0, ImageHeight - WindowSize + 1, Stride)
    XPositions = range(0, ImageWidth  - WindowSize + 1, Stride)

    for YPosition in YPositions:
        for XPosition in XPositions:
            OccludedArray = ImageArray.copy()
            MeanColor = ImageArray[YPosition : YPosition + WindowSize, XPosition : XPosition + WindowSize,].mean(axis=(0, 1)).astype(np.uint8)
            OccludedArray[
                YPosition : YPosition + WindowSize,
                XPosition : XPosition + WindowSize,
            ] = MeanColor

            OccludedImage      = Image.fromarray(OccludedArray)
            OccludedConfidence = GetConfidence(OccludedImage)
            ConfidenceDrop     = BaselineConfidence - OccludedConfidence

            HeatMap[YPosition : YPosition + WindowSize,
                    XPosition : XPosition + WindowSize] += ConfidenceDrop
            Counts[ YPosition : YPosition + WindowSize,
                    XPosition : XPosition + WindowSize] += 1

    Counts[Counts == 0] = 1
    HeatMap /= Counts

    return HeatMap, BaselineConfidence