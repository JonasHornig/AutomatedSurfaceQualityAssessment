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
        self.PreparedData = {"Names": [], "Labels": [], "Indices": [], "FeatureTensor": torch.empty((1, 1), dtype=torch.float32)}
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
                    self.PreparedData["Names"].append(ActiveName)
                    self.PreparedData["Labels"].append(Label)
                    self.PreparedData["Indices"].append(LabelIndex)

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
    
    
    
    ValidationHandler.PreparedData["Distances"]       = [Distance.item() for Distance in DistanceMatrix.min(1)[0]]
    ValidationHandler.PreparedData["Probabilities"]   = [Probability.item() for Probability in ProbabilityMatrix.max(1)[0]]
    ValidationHandler.PreparedData["Classifications"] = [Classification.item() for Classification in DistanceMatrix.min(1)[1]]


    #some comment

    breakpoint()

    for Index in range(len(ValidationHandler.PreparedData["Names"])):
        print("")


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