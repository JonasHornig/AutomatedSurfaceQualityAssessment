import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import math

from . import Logging as LOG

# The DINOv3 model code is loaded from GitHub through torch.hub. It is pinned to a fixed commit,
# so later changes to the DINOv3 repository can't change the extracted features.
DINOV3_HUB_REPOSITORY = "facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751"

# Supported feature extractors. The name is used for the feature files <DataSet>_<Sample>_<Name>Features.h5 (it must not contain "_").
# The weights are gated and have to be downloaded from Meta into src/Models/.
DINOV3_MODELS = {
    # Name             : (torch.hub entry point, weights file)                                   # Parameters, feature dimension
    "DINOv3"           : ("dinov3_vitb16"    , "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"    ), #  86 M,  768
    "DINOv3-H16plus"   : ("dinov3_vith16plus", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"), # 840 M, 1280
}

class DINOv3Model():
    def __init__(self, Name: str, ModelDirectory: str):
        if Name not in DINOV3_MODELS:
            raise ValueError(f"Unknown feature extractor {Name}. Available: {list(DINOV3_MODELS)}")
        HubEntryPoint, WeightsName = DINOV3_MODELS[Name]
        self.Preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) , ]) # normalizing images the same way the DINOv3 (LVD-1689M) training images were normalized
        self.Model = torch.hub.load(DINOV3_HUB_REPOSITORY, HubEntryPoint, weights=f"{ModelDirectory}/{WeightsName}")

        assert isinstance(self.Model, nn.Module)
        self.Model.eval()

    def ApplyNetwork(self, Images, Log, ProgressBar=None, DetailedLog=False, MaxBatchSize: int = 64): # usage?
        """
        Extracts the DINOv3 features of one class and stores them as Images["DinoFeatures"] [Number of images, 768].
        Images["Images"] - list of PIL images
        Images["Names"]  - list of the corresponding image names, in the same order
        Optional tqdm progress bar
        """
        if not Images["Images"] or len(Images["Images"]) != len(Images["Names"]):
            raise ValueError(f"Expected the same, non-zero number of images and names, got {len(Images["Images"])} images and {len(Images["Names"])} names")

        BatchSize, NumberOfBatches = ComputeBatchSize(NumberOfImages=len(Images["Images"]), MaxBatchSize=MaxBatchSize)
        Log.W(f"Maximum Batch Size: {MaxBatchSize}, Calculated Batch Size: {BatchSize}, Resulting Number of Batches: {NumberOfBatches}")

        ExtractedFeatures: list[torch.Tensor] = []
        for BatchIndex in range(NumberOfBatches):
            BatchedImages = Images["Images"][BatchIndex*BatchSize:(BatchIndex+1)*BatchSize]
            BatchedNames  = Images["Names" ][BatchIndex*BatchSize:(BatchIndex+1)*BatchSize]

            PreprocessedBatch = torch.stack([self.Preprocessor(Entry) for Entry in BatchedImages])
            with torch.no_grad():
                BatchFeatures = self.Model(PreprocessedBatch) # type: ignore  --  Shape: [BatchSize, 768]

            if DetailedLog:
                LOG.FeatureBatch(Log, BatchIndex, NumberOfBatches, BatchedNames, BatchFeatures)
            ExtractedFeatures.append(BatchFeatures)

            if ProgressBar:
                ProgressBar.update(len(BatchedImages))

        Images["DinoFeatures"] = torch.cat(ExtractedFeatures, dim=0)  # Shape: [NumberOfImages, 768]

class ProtoNet(): # usage?
    def __init__(self, Log, Controls, FeatureDimension: int):
        Log.W("\nCreating Proto Net\n====================")

        self.NumberOfHiddenLayers = Controls.NumberOfHiddenLayers
        self.FeatureDimension     = FeatureDimension
        self.EmbeddingDimension   = Controls.EmbeddingDimension
        self.Gamma                = Controls.Gamma

        self.Encoder = self.GetMlpEncoder(Log)
        for Line in str(self.Encoder).split("\n"):
            Log.W(f"{Line}")

        self.Prototypes = torch.Tensor()

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
    
    def CalculateLoss(self, Episode):
        '''
        Loss J of one training episode, following Algorithm 1 from Snell et al. (2017).
        The prototype of each class is the mean embedding of its support samples. Each query sample is assigned the
        probability softmax(-d) over its squared Euclidean distances d to the prototypes, and the loss is the mean
        negative log probability of the correct class (= cross entropy with -d as logits).
        '''
        SupportTensor    = Episode["SupportTensor"]    # S_k, Dim: N_C, N_S, Feature dimension
        QueryTensor      = Episode["QueryTensor"]      # Q_k, Dim: N_C, N_Q, Feature dimension
        ValidationTensor = Episode["ValidationTensor"] # Held back validation images, Dim: Number of validation images, Feature dimension
        ValidationLabels = Episode["ValidationLabels"] # Episode label of each validation image
        NumberOfClasses, NumberOfSupportSamples, _ = SupportTensor.shape
        NumberOfQuerySamples                       = QueryTensor.size(1)

        # Support and query samples are embedded in one pass, so BatchNorm normalises them with the same batch statistics
        Embeddings        = self.Encoder(torch.cat([SupportTensor.flatten(0, 1), QueryTensor.flatten(0, 1)]))
        EpisodePrototypes = Embeddings[:NumberOfClasses * NumberOfSupportSamples].view(NumberOfClasses, NumberOfSupportSamples, -1).mean(1)
        QueryEmbeddings   = Embeddings[NumberOfClasses * NumberOfSupportSamples:]
        QueryLabels       = torch.arange(NumberOfClasses).repeat_interleave(NumberOfQuerySamples) # The query samples are ordered by class

        Distances = EuclideanDistance(QueryEmbeddings, EpisodePrototypes)
        Loss      = F.cross_entropy(-Distances, QueryLabels) # J
        Accuracy  = (Distances.argmin(1) == QueryLabels).float().mean()

        # Validation is done in evaluation mode (BatchNorm uses its running statistics) and without tracking gradients,
        # so that it neither influences the training nor depends on the composition of the batch.
        # The support samples are embedded again in evaluation mode, so that prototypes and validation samples are normalised the same way.
        self.Encoder.eval()
        with torch.no_grad():
            ValidationPrototypes = self.Encoder(SupportTensor.flatten(0, 1)).view(NumberOfClasses, NumberOfSupportSamples, -1).mean(1)
            ValidationOutput     = self.Encoder(ValidationTensor)
        self.Encoder.train()

        ValidationAccuracy = (EuclideanDistance(ValidationOutput, ValidationPrototypes).argmin(1) == ValidationLabels).float().mean()

        return Loss, {
            "Loss": Loss.item(),
            "Accuracy": Accuracy.item(),
            "ValidationAccuracy": ValidationAccuracy.item()}

    def ComputePrototypes(self, Images, LabelMap, Mode: str = "Training"): # usage?
        '''
        Computes the final class prototypes after training as the mean embedding of the images of a class.
        Indices {<Sample_Class> : image indices} limits this to the given images, e.g. excluding the held back validation images. Without it all images are used.
        The prototypes are ordered by the label map, so prototype k belongs to the class with label k.
        The prototypes of the individual episodes can't be used for this, as their order follows the random class selection of the episode.
        '''
        WasTraining = self.Encoder.training
        self.Encoder.eval()
        Prototypes = []
        with torch.no_grad():
            for CombinedClass, _ in sorted(LabelMap.items(), key=lambda Entry: Entry[1]):
                Sample, Class = CombinedClass.split("_", 1)
                Features   = Images[Sample][Class][Mode]["DinoFeatures"]
                Embeddings = self.Encoder(Features)
                Prototypes.append(Embeddings.mean(0))
        self.Encoder.train(WasTraining)
        self.Prototypes = torch.stack(Prototypes) # Dim: NumberOfClasses, EmbeddingDimension

    def SaveNetwork(self, OutputPath: str):
        # The prototypes are ordered by the label map, which is saved separately as Labelmap.json
        torch.save(self.Encoder.state_dict(), f"{OutputPath}/TrainedProtoNet.pth")
        torch.save(self.Prototypes, f"{OutputPath}/PrototypeEmbedding.pth")

def ComputeBatchSize(NumberOfImages: int, MaxBatchSize: int) -> tuple[int, int]:
    NumberOfBatches = math.ceil(NumberOfImages / MaxBatchSize)
    EvenBatchSize   = math.ceil(NumberOfImages / NumberOfBatches)
    return int(EvenBatchSize), int(NumberOfBatches)

def EuclideanDistance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Squared Euclidean distance between every row of x [n, d] and every row of y [m, d], Dim: n, m
    # (Computed directly instead of with torch.cdist, whose square root has no gradient at a distance of zero)
    return (x[:, None, :] - y[None, :, :]).pow(2).sum(-1)