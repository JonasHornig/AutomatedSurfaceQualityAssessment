import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import math

from . import AuxiallryCode as AUX
from . import Logging as LOG

class DINOv2Model():
    def __init__(self):
        self.Preprocessor = transforms.Compose([
            transforms.ToTensor()                                                           ,
            transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) , ]) # normalizing images the same way the DINO training images were normalized
        self.Model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.LoadModel()

    def LoadModel(self):
        assert isinstance(self.Model, nn.Module)
        self.Model.eval()

    def ApplyNetwork(self, Images, Log, ProgressBar = None, DetailedLog = False):
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

        self.Prototype = None

    def GetMlpEncoder(self, Log):
        @staticmethod
        def MlpBlock(InputDimension, OutputDimension):
            return nn.Sequential(
                nn.Linear(InputDimension, OutputDimension) ,
                nn.BatchNorm1d(OutputDimension)            ,
                nn.LeakyReLU()                             )
    
        NumberOfLayers = self.NumberOfHiddenLayers + 2
        Indices    = np.arange(0, NumberOfLayers+1, 1)
        Fraction   = (np.exp(-self.Gamma * Indices) - np.exp(-self.Gamma * NumberOfLayers)) / (1 - np.exp(-self.Gamma * NumberOfLayers))
        Dimensions = (self.EmbeddingDimension + (self.FeatureDimension-self.EmbeddingDimension)*Fraction).astype(int)

        MlpLayers = []
        for Index, _ in enumerate(Dimensions[:-1]):
            MlpLayers.append(MlpBlock(Dimensions[Index], Dimensions[Index+1]))

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
        #768 is the DINOv2 Output dimension
        #NumberOfClasses, NumberOfSamples, 768 --> NumberOfClasses * NumberOfSamples, 768
        ReshapedPreprocessedSupportTensor  = PreprocessedSupportTensor.view(  NumberOfClasses * NumberOfSupportSamples  , *PreprocessedSupportTensor.size()[2:])
        ReshapedPreprocessedQueryTensor    = PreprocessedQueryTensor.view(    NumberOfClasses * NumberOfQuerySamples    , *PreprocessedQueryTensor.size()[2:])
        ReshapedPreprocessedValidateTensor = PreprocessedValidateTensor.view( NumberOfClasses * NumberOfValidateSamples , *PreprocessedValidateTensor.size()[2:])
        #Connect to NumberOfClasses * NumberOfSupportSamples + NumberOfClasses * NumberOfSupportSamples, 768
        InputTensor = torch.cat([ReshapedPreprocessedSupportTensor, ReshapedPreprocessedQueryTensor], 0)

        self.OutputTensor     = self.Encoder.forward( InputTensor                        )
        self.ValidationOutput = self.Encoder.forward( ReshapedPreprocessedValidateTensor )

        OutputDimensions = self.OutputTensor.size(-1)
        QueryEmbeddings = self.OutputTensor[NumberOfClasses * NumberOfSupportSamples:]
        self.Prototype = self.OutputTensor[:NumberOfClasses * NumberOfSupportSamples].view(NumberOfClasses,NumberOfSupportSamples,OutputDimensions).mean(1)

        '''
        log_softmax = (z_i) = log(e^(z_i)/sum(e^(z_j))) = z_i - log(sum(e^(z_j)))
        with -DistanceMatrix:
        d_(ik) is the distance of the i-th query vector to the k-th prototype
        log_softmax( -d_(ik) ) = -d_(ik) - log(sum^(N_C)_(k'=1)(e^(-d_(ik'))))
        '''

        DistanceMatrix    = EuclideanDistance(QueryEmbeddings, self.Prototype)
        ProbabilityMatrix = F.log_softmax(-DistanceMatrix, dim=1).view(NumberOfClasses, NumberOfQuerySamples, -1)
        Loss              = -ProbabilityMatrix.gather(2, TargetIndices).squeeze().view(-1).mean()                 # J

        _, ResultLabel = ProbabilityMatrix.max(2)
        Accuracy       = torch.eq(ResultLabel.squeeze(), TargetIndices.squeeze()).float().mean()

        ValidationDistanceMatrix = EuclideanDistance(self.ValidationOutput, self.Prototype)
        _, ValidationResultLabel = ValidationDistanceMatrix.min(1)
        ValidationAccuracy       = torch.eq(ValidationResultLabel.squeeze(), TargetValidationIndices.squeeze()).float().mean()

        return Loss, {
            "Loss": Loss.item(),
            "Accuracy": Accuracy.item(),
            "ValidationAccuracy": ValidationAccuracy.item()}


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