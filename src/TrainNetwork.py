import time
StartTime = time.time()

import torch
import os
import random
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass, field

import Modules.DataSetHandling as DATA
import Modules.NeuralNetwork as NN
import Modules.Logging as LOG
import Modules.EpisodeHandler as EPH
import Modules.NetworkValidation as VAL

@dataclass
class ControlVariables:
    # Inputs
    Modes      : tuple = ("Training", "Testing")
    DataSet    : str   = "CompleteDataSet" #  --  DummyDataSet  --  SmallDataSet  --  CompleteDataSet
    FeatureExtractor : str = "DINOv3"   #  --  DINOv3  --  legacy was DINOv1 ViT-B/16

    FeatureDimension     : int   = 768
    EmbeddingDimension   : int   = 5
    NumberOfHiddenLayers : int   = 4
    Gamma                : float = 0.75

    NumberOfEpisodes               : int   = 500
    LearningRate                   : float = 1e-4
    WeightDecay                    : float = 1e-2  # AdamW weight decay (1e-2 is the PyTorch default)
    UseLearningRateScheduler       : bool  = False # Cosine annealing of the learning rate from LearningRate to 0 over NumberOfEpisodes. (slightly worse results than a constant learning rate, but the final validation accuracy is more stable)
    Seed                           : int   = 333     # Seed for all random number generators, makes training runs reproducible
    NumberOfClassesPerEpisode      : int   = 6     # N_C =< K
    NumberOfSupportSamplesPerClass : int   = 9     # N_S
    NumberOfQuerySamplesPerClass   : int   = 1     # N_Q
    ValidationFraction             : float = 0.25  # Fraction of the training images of each class held back for validation, never used for support or query samples

    # Flags
    #  --  True  --  False  --
    WriteDetailedDebugInfo : bool = True
    
    # Parameter initialisation
    NumberOfClasses : int = 0 # K
    
    def __post_init__(self):
        BaseDirectory = os.path.dirname(os.path.abspath(__file__))
        self.DataSetPath = f"{BaseDirectory}/DataSets/{self.DataSet}"
        self.OutputPath : str   = f"{BaseDirectory}/Outputs/"

def SetSeed(Seed: int):
    # Episode sampling uses Python's random module and torch, the weight initialisation uses torch
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)

def Main():
    Controls  = ControlVariables()
    SetSeed(Controls.Seed)
    print("")
    LogFile = LOG.NoteFile(f"{Controls.OutputPath}/Training.log")
    LogFile.W("\n****************************\n*  Training the Proto Net  *\n****************************")
    DataSet = DATA.DataSet()
    DataSet.LoadDataSet(LogFile, Controls) # Stops with an error if the features have not been extracted with PreprocessImages.py yet

    ProtoNet  = NN.ProtoNet(LogFile, Controls)
    Optimizer = torch.optim.AdamW(ProtoNet.Encoder.parameters(), lr=Controls.LearningRate, weight_decay=Controls.WeightDecay)
    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=Controls.NumberOfEpisodes)

    EpisodeLoader = EPH.EpisodeLoader(DataSet.Images, DataSet.GetParameters(), Controls)
    LOG.Training(Controls, EpisodeLoader, LogFile)
    LOG.ValidationSplit(LogFile, EpisodeLoader, Controls.WriteDetailedDebugInfo)
    EpisodeLoader.GenerateLabelMap(Controls.OutputPath)
    LOG.WritelabelMap(LogFile, EpisodeLoader.LabelMap)

    RunningLoss               = []
    RunningAccuracy           = []
    RunningValidationAccuracy = []

    print("Main training loop")
    ProtoNet.Encoder.train()
    for EpisodeIndex in tqdm(range(1, Controls.NumberOfEpisodes + 1), desc="Progress", unit="ep"):
        if Controls.WriteDetailedDebugInfo:
            LogFile.W(f"\nEpisode {EpisodeIndex:>5} out of {Controls.NumberOfEpisodes:>5}")

        ActiveEpisode = EpisodeLoader.GetRandomEpisode(LogFile, Controls)
        Optimizer.zero_grad() # Gradients have to be reset each episode, otherwise they accumulate over all previous episodes
        Loss, LossInfo = ProtoNet.CalculateLoss(ActiveEpisode)

        RunningLoss.append(LossInfo["Loss"])
        RunningAccuracy.append(LossInfo["Accuracy"])
        RunningValidationAccuracy.append(LossInfo["ValidationAccuracy"])

        Loss.backward()
        Optimizer.step()
        if Controls.UseLearningRateScheduler:
            Scheduler.step()

    # The final prototypes are computed from the training images only, so the held back validation images stay unseen
    ProtoNet.ComputePrototypes(DataSet.Images, EpisodeLoader.LabelMap, EpisodeLoader.TrainingIndices)
    ProtoNet.SaveNetwork(LogFile, Controls.OutputPath)

    ValidationFeatures, ValidationLabels = EpisodeLoader.GetValidationSet()
    FinalValidationAccuracy = ProtoNet.EvaluateAccuracy(ValidationFeatures, ValidationLabels)
    Message = f"Final validation accuracy (final prototypes, {len(ValidationLabels)} held back training images): {100*FinalValidationAccuracy:.3f} %"
    print(Message)
    LogFile.W(f"\n{Message}")
    LOG.PrintTrainingProcess(LogFile, Controls, RunningLoss, RunningAccuracy, RunningValidationAccuracy)

    ValidationHandler = VAL.ValidationHandler(DataSet, DataSet.GetParameters(), EpisodeLoader, ProtoNet)
    VAL.NetworkValidationMain(LogFile, Controls, ProtoNet, ValidationHandler)

    Duration = time.time() - StartTime
    Hours = int(Duration // 3600)
    Minutes = int((Duration % 3600) // 60)
    Seconds = int(Duration % 60)
    print(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}")
    LogFile.W(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}", NewLine=False)
    LogFile.Close()


if __name__ == "__main__":
    Main()