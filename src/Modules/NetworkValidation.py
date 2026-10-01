import torch

from . import Logging as LOG
from . import NeuralNetwork as NN

class ValidationHandler():
    def __init__(self, DataSet, DataSetParameters, EpisodeLoader, Network):
        self.Samples     : list[str] = DataSetParameters["Samples"]
        self.Classes     : list[str] = DataSetParameters["Classes"]
        self.Images      : dict      = DataSet.Images

        self.LabelMap    : dict      = EpisodeLoader.LabelMap
        self.LookUpTable : dict      = EpisodeLoader.LookUpTable

        self.DataPrepared : bool = False

        self.PostInit()

    def PostInit(self):
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]

    def ExtractData(self, LogFile, Controls, Mode="Testing"):
        # Generate training tensors with the shape [NC, NS, DINOv2 Feature dimension]
        self.PreparedData = {"MasterIndices": [], "Names": [], "Labels": [], "LabelIndices": [], "FeatureTensor": torch.empty((1, 1), dtype=torch.float32)}
        if Controls.WriteDetailedDebugInfo:
            LogFile.W("\nExtracting data from Data set for the test application")
            LogFile.W("Read data:\nIndex Name                                               Label                Index Tensor")
        RunningIndex = 0
        ClassTensors = []
        for Sample in self.Samples:
            for Class in self.Classes:
                Label      = f"{Sample}_{Class}"
                LabelIndex = self.LabelMap[Label]
                Features = []
                for Index in range(self.Images[Sample][Class]["Testing"]["DinoFeatures"].shape[0]):
                    ActiveFeatureTensor = self.Images[Sample][Class]["Testing"]["DinoFeatures"][Index]
                    ActiveName = self.Images[Sample][Class]["Testing"]["Names"][Index]

                    Features.append(ActiveFeatureTensor)
                    self.PreparedData["MasterIndices"].append(RunningIndex)
                    self.PreparedData["Names"].append(ActiveName)
                    self.PreparedData["Labels"].append(Label)
                    self.PreparedData["LabelIndices"].append(LabelIndex)

                    if Controls.WriteDetailedDebugInfo:
                        LOG.DataExtraction_One(LogFile, ActiveName, ActiveFeatureTensor,Label, LabelIndex, RunningIndex)
                    RunningIndex += 1
                ClassTensors.append(torch.stack(Features))

        self.PreparedData["FeatureTensor"] = torch.stack(ClassTensors)
        if Controls.WriteDetailedDebugInfo:
            LOG.DataExtraction_Two(LogFile, self.PreparedData, self.Samples, self.Classes)
        self.DataPrepared = True

def NetworkValidationMain(LogFile, Controls, ProtoNet, ValidationHandler):
    LogFile.W("\n******************************\n*  Validating the Proto Net  *\n******************************")
    print("\nValidating the trained network")
    NetworkTestApplication(LogFile, Controls, ProtoNet, ValidationHandler)

def NetworkTestApplication(LogFile, Controls, Network, ValidationHandler):
    ValidationHandler.ExtractData(LogFile, Controls, "Testing")
    Embeddings         = Network.Encoder.forward(ValidationHandler.PreparedData["FeatureTensor"].view(6 * 10, 768))
    DistanceMatrix     = NN.EuclideanDistance(Embeddings, Network.Prototypes)
    ProbabilityMatrix  = torch.nn.functional.softmax(-DistanceMatrix, dim=1)
    Classifications = {"DistanceBased"  : DistanceMatrix.min(1) , "ProbabilityBased"  : ProbabilityMatrix.max(1) }

    ValidationHandler.PreparedData["Distances"]       = []
    ValidationHandler.PreparedData["Probabilities"]   = []
    ValidationHandler.PreparedData["Classifications"] = []
    for Index in ValidationHandler.PreparedData["MasterIndices"]:
        ValidationHandler.PreparedData["Distances"].append(DistanceMatrix.min(1)[0][Index].item())
        ValidationHandler.PreparedData["Probabilities"].append(ProbabilityMatrix.max(1)[0][Index].item())
        ValidationHandler.PreparedData["Classifications"].append(DistanceMatrix.min(1)[1][Index].item())

    LOG.TestApplicationResults(LogFile, ValidationHandler.PreparedData, ValidationHandler.LookUpTable)