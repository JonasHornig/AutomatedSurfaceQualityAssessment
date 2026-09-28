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

        self.PostInit()

    def PostInit(self):
        self.CombinedClasses = [f"{Sample}_{Class}" for Sample in self.Samples for Class in self.Classes]

    def ExtractData(self, LogFile, Controls, Mode="Testing"):
        self.PreparedData = {"Features": [], "Names": [], "Labels": [], "Indices": []}
        if Controls.WriteDetailedDebugInfo:
            LogFile.W("\nExtracting data from Data set for the test application")
            LogFile.W("Read data:\nIndex Name                                               Label                Index Tensor")
        RunningIndex = 0
        for Sample in self.Samples:
            for Class in self.Classes:
                Label = f"{Sample}_{Class}"
                for Index in range(self.Images[Sample][Class]["Testing"]["DinoFeatures"].shape[0]):
                    self.PreparedData["Features"].append(self.Images[Sample][Class]["Testing"]["DinoFeatures"][Index])
                    self.PreparedData["Names"].append(self.Images[Sample][Class]["Testing"]["Names"][Index])
                    self.PreparedData["Labels"].append(Label)
                    self.PreparedData["Indices"].append(self.LabelMap[Label])
                    if Controls.WriteDetailedDebugInfo:
                        LOG.DataExtraction_One(LogFile, self.Images[Sample][Class]["Testing"]["Names"][Index], self.Images[Sample][Class]["Testing"]["DinoFeatures"][Index], Label, self.LabelMap[Label], RunningIndex)
                    RunningIndex += 1
        if Controls.WriteDetailedDebugInfo:
            LOG.DataExtraction_Two(LogFile, self.PreparedData)

def NetworkValidationMain(LogFile, Controls, ProtoNet, ValidationHandler):
    LogFile.W("\n******************************\n*  Validating the Proto Net  *\n******************************")
    print("\nValidating the trained network")
    NetworkTestApplication(LogFile, Controls, ProtoNet, ValidationHandler)

def NetworkTestApplication(LogFile, Controls, Network, ValidationHandler):
    ValidationHandler.ExtractData(LogFile, Controls, "Testing")

    '''
    TensorsToStack = []
    for Image in Images:
        Image.ImageTensor = self.Transform(Image.PILImage)
        TensorsToStack.append(Image.ImageTensor)
    StackedTensors = torch.stack(TensorsToStack)

    with torch.no_grad():
        Preprocessed       = self.PretrainedNet[1](StackedTensors)
        PreprocessedTensor = self.PretrainedNet[0](Preprocessed)
        EmbeddedImage      = self.TrainedProtoNet.encoder.forward(PreprocessedTensor)
        DistanceMatrix     = EuclideanDistance(EmbeddedImage, self.PrototypeEmbedding)
        ProbabilityMatrix  = torch.nn.functional.softmax(-DistanceMatrix, dim=1)

    Matrices        = {"DistanceMatrix" : DistanceMatrix        , "ProbabilityMatrix" : ProbabilityMatrix        }
    Classifications = {"DistanceBased"  : DistanceMatrix.min(1) , "ProbabilityBased"  : ProbabilityMatrix.max(1) }

    if len(Images) != Classifications["DistanceBased"][0].shape[0]:
        print("Images got lost in network application")
        print(f"Number of Image tensors: {Classifications["DistanceBased"][0].shape[0]}")
        print(f"Number of loaded images : {len(Images)}")
        raise SystemExit()

    for Index, Image in enumerate(Images):
        if not torch.equal(StackedTensors[Index], Image.ImageTensor):
            print("Error! Image indexing inconsistent - See Network application")
            print(f"Occurred for Index {Index}")
            raise SystemExit()
        if Classifications["DistanceBased"][1][Index].item() == Classifications["ProbabilityBased"][1][Index].item():
            Image.CoherentClassification = True

        ClassificationID        = Classifications["DistanceBased"][1][Index].item()
        Image.NNClassification  = [ClassificationID, self.LookUpTable[ClassificationID]]
        Image.LookUpTable       = self.LookUpTable
        Image.Distances         = Matrices["DistanceMatrix"][Index]
        Image.Probabilities     = Matrices["ProbabilityMatrix"][Index]
        Image.PreprocessedImage = PreprocessedTensor[Index]
        Image.EmbeddedImage     = EmbeddedImage[Index]

    with torch.no_grad():
        # StackedTensors are features -> pass to encoder
        EmbeddedFeatures = Network.encoder.forward(StackedTensors) 
        
        # Calculate distances to the loaded prototypes
        DistanceMatrix = NN.EuclideanDistance(EmbeddedFeatures, Network.Prototypes)
        
        # Get the index of the closest prototype
        _, Classifications_idx = DistanceMatrix.min(1)
    '''