import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import math

from . import AuxiliaryCode as AUX
from . import Logging as LOG

class DINOv3Model():
    def __init__(self, WeightsPath: str):
        self.Preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) , ]) # normalizing images the same way the DINOv3 (LVD-1689M) training images were normalized
        # DINOv3 weights are gated - request access at ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and download the ViT-B/16 checkpoint
        self.Model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16', weights=WeightsPath)
        self.LoadModel()

    def LoadModel(self):
        assert isinstance(self.Model, nn.Module)
        self.Model.eval()

    def ApplyNetwork(self, Images, Log, ProgressBar=None, DetailedLog=False):
        """
        Expects data as python dictionary (Images)
        Images["Images"] - list of images
          Loaded in testing with PIL
        Images["Names"] - list of corresponding image names
          It is important that both lists have the same order
          From now on this will be assumed to be true
        
        Optional tqdm progress bar
        """
        
        AUX.ConfirmEqualLength(Images["Images"], Images["Names"])
        
        MaxBatchSize = 64
        BatchSize, NumberOfBatches = ComputeBatchSize(NumberOfImages=len(Images["Images"]), MaxBatchSize=MaxBatchSize)
        LOG.ApplyDinoNetwork_FirstInstance(Log, MaxBatchSize, BatchSize, NumberOfBatches, DetailedLog, Images)

        ExtractedFeatures: list[torch.Tensor] = []
        for BatchIndex in range(NumberOfBatches):
            if DetailedLog:
                Log.W(f"        Batch Number {BatchIndex+1} out of {NumberOfBatches}")
                Log.W(f"        Index Name                           Dimension   Feature Vector")
            BatchedImages = Images["Images"][BatchIndex*BatchSize:(BatchIndex+1)*BatchSize]
            BatchedNames  = Images["Names" ][BatchIndex*BatchSize:(BatchIndex+1)*BatchSize]
            
            PreprocessedBatch = torch.stack([self.Preprocessor(Entry) for Entry in BatchedImages])
            with torch.no_grad():
                BatchFeatures = self.Model(PreprocessedBatch) # type: ignore  --  Shape: [BatchSize, 768]

            LOG.ApplyDinoNetwork_SecondInstance(Log, DetailedLog, BatchedImages, BatchFeatures, BatchedNames)
            
            ExtractedFeatures.append(BatchFeatures)
            
            if ProgressBar:
                ProgressBar.update(len(BatchedImages))
        
        FeatureTensor = torch.cat(ExtractedFeatures, dim=0)  # Shape: [NumberOfImages, 768]
        Images["DinoFeatures"] = FeatureTensor

        LOG.ApplyDinoNetwork_ThirdInstance(DetailedLog, Log, FeatureTensor, Images)

class ProtoNet():
    def __init__(self, Log, Controls):
        Log.W("\nCreating Proto Net\n====================")

        self.NumberOfHiddenLayers = Controls.NumberOfHiddenLayers
        self.FeatureDimension     = Controls.FeatureDimension
        self.EmbeddingDimension   = Controls.EmbeddingDimension
        self.Gamma                = Controls.Gamma

        self.Encoder = self.GetMlpEncoder(Log)
        for Line in str(self.Encoder).split("\n"):
            Log.W(f"{Line}")

        self.Prototypes = None

    def GetMlpEncoder(self, Log):
        @staticmethod
        def MlpBlock(InputDimension, OutputDimension):
            return nn.Sequential(
                nn.Linear(InputDimension, OutputDimension) ,
                nn.BatchNorm1d(OutputDimension)            ,
                nn.LeakyReLU()                             )
    
        # The input layer is the DINOv3 feature vector itself. It is followed by NumberOfHiddenLayers hidden blocks (Linear, BatchNorm, LeakyReLU)
        # and a purely linear output layer projecting into the embedding space, so the embedding distances are not distorted by an activation.
        # Dimensions: Feature -> NumberOfHiddenLayers hidden widths -> Embedding
        NumberOfLayers = self.NumberOfHiddenLayers + 1
        Indices    = np.arange(0, NumberOfLayers+1, 1)
        Fraction   = (np.exp(-self.Gamma * Indices) - np.exp(-self.Gamma * NumberOfLayers)) / (1 - np.exp(-self.Gamma * NumberOfLayers))
        Dimensions = (self.EmbeddingDimension + (self.FeatureDimension-self.EmbeddingDimension)*Fraction).astype(int)

        MlpLayers = []
        for Index in range(self.NumberOfHiddenLayers):
            MlpLayers.append(MlpBlock(Dimensions[Index], Dimensions[Index+1]))
        MlpLayers.append(nn.Linear(Dimensions[-2], Dimensions[-1]))

        LOG.GenerateEncoder(Log, self.FeatureDimension, self.EmbeddingDimension, self.NumberOfHiddenLayers, self.Gamma, Indices, Dimensions)
        return nn.Sequential(*MlpLayers)
    
    def CalculateLoss(self, sample):
        PreprocessedSupportTensor = Variable(sample["SupportTensor"])   # S_k
        PreprocessedQueryTensor = Variable(sample["QueryTensor"])       # Q_k
        PreprocessedValidateTensor = Variable(sample["ValidationTensor"])

        NumberOfClasses = PreprocessedSupportTensor.size(0)          # N_C ~~ K
        NumberOfSupportSamples = PreprocessedSupportTensor.size(1)   # N_S
        NumberOfQuerySamples = PreprocessedQueryTensor.size(1)       # N_Q
        NumberOfValidateSamples = PreprocessedValidateTensor.size(1)

        TargetIndices = torch.arange(0, NumberOfClasses).view(NumberOfClasses, 1, 1).expand(NumberOfClasses,NumberOfQuerySamples,1).long()
        # Dim: NumberOfClasses, NumberOfQuerySamples, 1
        TargetIndices           = Variable(TargetIndices, requires_grad=False)
        TargetValidationIndices = torch.arange(0, NumberOfClasses).repeat_interleave(NumberOfValidateSamples)
        # Dim: NumberOfClasses * NumberOfValidateSamples

        if PreprocessedQueryTensor.is_cuda:
            TargetIndices = TargetIndices.cuda()

        #Reshape dimensions to fit the network/embedding function:
        #768 is the DINOv3 ViT-B/16 Output dimension
        #NumberOfClasses, NumberOfSamples, 768 --> NumberOfClasses * NumberOfSamples, 768
        ReshapedPreprocessedSupportTensor  = PreprocessedSupportTensor.view(  NumberOfClasses * NumberOfSupportSamples  , *PreprocessedSupportTensor.size()[2:])
        ReshapedPreprocessedQueryTensor    = PreprocessedQueryTensor.view(    NumberOfClasses * NumberOfQuerySamples    , *PreprocessedQueryTensor.size()[2:])
        ReshapedPreprocessedValidateTensor = PreprocessedValidateTensor.view( NumberOfClasses * NumberOfValidateSamples , *PreprocessedValidateTensor.size()[2:])
        #Connect to NumberOfClasses * NumberOfSupportSamples + NumberOfClasses * NumberOfSupportSamples, 768
        InputTensor = torch.cat([ReshapedPreprocessedSupportTensor, ReshapedPreprocessedQueryTensor], 0)

        self.OutputTensor     = self.Encoder.forward( InputTensor                        )

        OutputDimensions = self.OutputTensor.size(-1)
        QueryEmbeddings = self.OutputTensor[NumberOfClasses * NumberOfSupportSamples:]
        EpisodePrototypes = self.OutputTensor[:NumberOfClasses * NumberOfSupportSamples].view(NumberOfClasses,NumberOfSupportSamples,OutputDimensions).mean(1)

        '''
        log_softmax = (z_i) = log(e^(z_i)/sum(e^(z_j))) = z_i - log(sum(e^(z_j)))
        with -DistanceMatrix:
        d_(ik) is the distance of the i-th query vector to the k-th prototype
        log_softmax( -d_(ik) ) = -d_(ik) - log(sum^(N_C)_(k'=1)(e^(-d_(ik'))))
        '''

        DistanceMatrix    = EuclideanDistance(QueryEmbeddings, EpisodePrototypes)
        ProbabilityMatrix = F.log_softmax(-DistanceMatrix, dim=1).view(NumberOfClasses, NumberOfQuerySamples, -1)
        Loss              = -ProbabilityMatrix.gather(2, TargetIndices).squeeze().view(-1).mean()                 # J

        _, ResultLabel = ProbabilityMatrix.max(2)
        Accuracy       = torch.eq(ResultLabel.squeeze(), TargetIndices.squeeze()).float().mean()

        # Validation is done in evaluation mode (BatchNorm uses its running statistics) and without tracking gradients,
        # so that it neither influences the training nor depends on the composition of the batch.
        # The support samples are embedded again in evaluation mode, so that prototypes and validation samples are normalised the same way.
        self.Encoder.eval()
        with torch.no_grad():
            ValidationPrototypes  = self.Encoder.forward(ReshapedPreprocessedSupportTensor).view(NumberOfClasses,NumberOfSupportSamples,OutputDimensions).mean(1)
            self.ValidationOutput = self.Encoder.forward(ReshapedPreprocessedValidateTensor)
        self.Encoder.train()

        ValidationDistanceMatrix = EuclideanDistance(self.ValidationOutput, ValidationPrototypes)
        _, ValidationResultLabel = ValidationDistanceMatrix.min(1)
        ValidationAccuracy       = torch.eq(ValidationResultLabel.squeeze(), TargetValidationIndices.squeeze()).float().mean()

        return Loss, {
            "Loss": Loss.item(),
            "Accuracy": Accuracy.item(),
            "ValidationAccuracy": ValidationAccuracy.item()}

    def ComputePrototypes(self, Images, LabelMap, Mode: str = "Training"):
        '''
        Computes the final class prototypes after training as the mean embedding of all images of a class.
        The prototypes are ordered by the label map, so prototype k belongs to the class with label k.
        The prototypes of the individual episodes can't be used for this, as their order follows the random class selection of the episode.
        '''
        WasTraining = self.Encoder.training
        self.Encoder.eval()
        Prototypes = []
        with torch.no_grad():
            for CombinedClass, _ in sorted(LabelMap.items(), key=lambda Entry: Entry[1]):
                Sample, Class = CombinedClass.split("_", 1)
                Embeddings = self.Encoder(Images[Sample][Class][Mode]["DinoFeatures"])
                Prototypes.append(Embeddings.mean(0))
        self.Encoder.train(WasTraining)
        self.Prototypes = torch.stack(Prototypes) # Dim: NumberOfClasses, EmbeddingDimension

    def SaveNetwork(self, Log, OutputPath: str):
        # The prototypes are ordered by the label map, which is saved separately as Labelmap.json
        torch.save(self.Encoder.state_dict(), f"{OutputPath}/TrainedProtoNet.pth")
        torch.save(self.Prototypes, f"{OutputPath}/PrototypeEmbedding.pth")
        Log.W(f"\nSaved the encoder weights to {OutputPath}TrainedProtoNet.pth and the prototypes {list(self.Prototypes.shape)} to {OutputPath}PrototypeEmbedding.pth")

    def ApplyNetwork(self):
        return

'''
class CompleteNetwork:
    def __init__(self, Settings):
        self.Settings = Settings

        self.LabelMap    = GLM.GetLabelMap()
        self.LookUpTable = {self.LabelMap[Class]: Class for Class in self.LabelMap.keys()}

        self.Transform       = self.GenerateTensor()
        self.PretrainedNet   = self.GetDINOv3(self.Settings.DinoWeightsPath)
        self.TrainedProtoNet = self.GetProtoNet()

    @staticmethod
    def GetDINOv3(WeightsPath):
        Preprocessor = transforms.Compose([
            transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) , ]) # Tiles are already 224 x 224, no cropping or resizing

        Features = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16', weights=WeightsPath)
        return [Features.eval(), Preprocessor]

    def GetProtoNet(self):
        if isinstance(self.Settings.NetworkPath, str):
            TrainedProtoNet = self.CreateLinearNet()
            TrainedProtoNet.load_state_dict(torch.load(self.Settings.NetworkPath + "/TrainedProtoNet.pth", weights_only=True))
            self.PrototypeEmbedding = torch.load(self.Settings.NetworkPath + "/PrototypeEmbedding.pth", weights_only=True)
        else:
            print("Can't load a network. See NeuralNetworkClassification.py - class LoadCompleteNetwork - GetProtoNet()")
            raise SystemExit()
        return TrainedProtoNet.eval()

    def CreateLinearNet(self):
        class Protonet(nn.Module):
            def __init__(self, encoder):
                super(Protonet, self).__init__()
                self.encoder = encoder

        if self.Settings.NumberOfLayers < 2:
            print("Setting number of layers to 2.")
            self.Settings.NumberOfLayers = 2

        def CreateLinearNetworkBlock(InputShape, OutputShape, LocalDropOutRate=0):
            LinearBlock = [
                nn.Linear(InputShape, OutputShape),
                nn.BatchNorm1d(OutputShape),
                nn.LeakyReLU()]
            if LocalDropOutRate > 0:
                LinearBlock.append(nn.Dropout(p=LocalDropOutRate))
            return nn.Sequential(*LinearBlock)

        ProgressionFactor = int(np.exp(np.log(self.Settings.PretrainedOutputDimension) / self.Settings.NumberOfLayers))
        ActiveLayerWidth = ProgressionFactor ** (self.Settings.NumberOfLayers - 1)
        Layers = [CreateLinearNetworkBlock(self.Settings.PretrainedOutputDimension, ActiveLayerWidth, self.Settings.DropOutRate)]
        for _ in range(self.Settings.NumberOfLayers - 2):
            NextLayerWidth = int(ActiveLayerWidth / ProgressionFactor)
            Layers.append(CreateLinearNetworkBlock(ActiveLayerWidth, NextLayerWidth, self.Settings.DropOutRate))
            ActiveLayerWidth = NextLayerWidth

        Layers.append(CreateLinearNetworkBlock(ActiveLayerWidth, self.Settings.EmbeddingDimensions))
        encoder = nn.Sequential(*Layers)

        if len(Layers) != self.Settings.NumberOfLayers:
            print(f"ERROR! {len(Layers)} created but {self.Settings.NumberOfLayers} layers requested")
            raise SystemExit
        return Protonet(encoder)

    def ApplyNetwork(self, Images):
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

            if Image.Probabilities.max(0)[0].item() < 0.9:
                Image.UnsureClassification = True

    @staticmethod
    def GenerateTensor():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
'''

def ComputeBatchSize(NumberOfImages: int, MaxBatchSize: int) -> tuple[int, int]:
    """
    Calculates an optimal batch size so that all batches are as even as possible
    Formula: BatchSize = ceil(N / ceil(N / MaxBatchSize))
    """
    NumberOfBatches = math.ceil(NumberOfImages / MaxBatchSize)
    EvenBatchSize   = math.ceil(NumberOfImages / NumberOfBatches)
    return int(EvenBatchSize), int(NumberOfBatches)

def EuclideanDistance(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)