import time
StartTime = time.time()

import torch
import os
import json
import random
import numpy as np

from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, asdict

import Modules.DataSetHandling as DATA
import Modules.NeuralNetwork as NN
import Modules.Logging as LOG
import Modules.EpisodeHandler as EPH
import Modules.NetworkTesting as TEST

@dataclass
class ControlVariables:
    # Inputs
    DataSet    : str   = "CompleteDataSet" #  --  DummyDataSet  --  SmallDataSet  --  CompleteDataSet
    FeatureExtractor : str = "DINOv3"   #  --  "DINOv3" (ViT-B/16)  --  "DINOv3-H16plus" (ViT-H+/16)  --  "DINOv1" (legacy, provided as files only)

    # Network. The input dimension is taken from the feature files (768 for DINOv3 ViT-B/16)
    EmbeddingDimension   : int   = 8 # 3 to 8 performed equally well in a study (5 seeds each), 8 gave the most consistent results
    NumberOfHiddenLayers : int   = 4
    Gamma                : float = 0.75

    NumberOfEpisodes               : int   = 250
    LearningRate                   : float = 1e-4
    WeightDecay                    : float = 1e-2  # AdamW weight decay (1e-2 is the PyTorch default)
    UseLearningRateScheduler       : bool  = False # Cosine annealing of the learning rate from LearningRate to 0 over NumberOfEpisodes. (slightly worse results than a constant learning rate, but the final validation accuracy is more stable)
    Seed                           : int   = 333   # Seed for all random number generators, makes training runs reproducible
    NumberOfClassesPerEpisode      : int   = 6     # N_C =< K
    NumberOfSupportSamplesPerClass : int   = 10    # N_S
    NumberOfQuerySamplesPerClass   : int   = 10    # N_Q

    # Outputs
    RunName : str = "" # Name of the output folder Outputs/<RunName>/. Empty: date and time of the start of the run, so runs never overwrite each other

    # Flags
    #  --  True  --  False  --
    WriteDetailedDebugInfo : bool = False     # Enabling this may severely impact performance
    GenerateRandomSeed     : bool = False

    def __post_init__(self):
        BaseDirectory = os.path.dirname(os.path.abspath(__file__))
        if not self.RunName:
            self.RunName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.DataSetPath : str = f"{BaseDirectory}/DataSets/{self.DataSet}"
        self.OutputPath  : str = f"{BaseDirectory}/Outputs/{self.RunName}"
        os.makedirs(self.OutputPath, exist_ok=True)

        self.SetSeed()
        self.SaveSettings()

    def SaveSettings(self):
        with open(f"{self.OutputPath}/Settings.json", "w") as SettingsFile:
            json.dump(asdict(self), SettingsFile, indent=4)

    def SetSeed(self):
        if self.GenerateRandomSeed:
            self.Seed = random.randint(100000000, 999999999)
        random.seed(self.Seed)
        np.random.seed(self.Seed)
        torch.manual_seed(self.Seed)

def Main():
    Controls  = ControlVariables()
    LogFile = LOG.NoteFile(f"{Controls.OutputPath}/Training.log")

    print(f"\nRun {Controls.RunName}\n=======================")
    LogFile.W("\n****************************\n*  Training the Proto Net  *\n****************************")
    LogFile.W(f"Run {Controls.RunName}")

    DataSet = DATA.DataSet()
    DataSet.LoadDataSet(LogFile, Controls)

    ProtoNet  = NN.ProtoNet(LogFile, Controls, DataSet.FeatureDimension)
    Optimizer = torch.optim.AdamW(ProtoNet.Encoder.parameters(), lr=Controls.LearningRate, weight_decay=Controls.WeightDecay)
    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=Controls.NumberOfEpisodes)

    EpisodeLoader = EPH.EpisodeLoader(DataSet, Controls)
    EpisodeLoader.Initialize(LogFile)
    EpisodeLoader.GenerateLabelMap(Controls.OutputPath)
    LOG.TrainingSettings(LogFile, Controls, EpisodeLoader)

    RunningLoss               = []
    RunningAccuracy           = []
    RunningValidationAccuracy = []

    print("\nMain training loop")
    ProtoNet.Encoder.train()
    for EpisodeIndex in tqdm(range(1, Controls.NumberOfEpisodes + 1), desc="Progress", unit="ep"):
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
    
    LOG.PrintTrainingProcess(LogFile, Controls, RunningLoss, RunningAccuracy, RunningValidationAccuracy)
    #ProtoNet.ComputePrototypes(DataSet.Images, EpisodeLoader.LabelMap, EpisodeLoader.TrainingIndices)
    ProtoNet.ComputePrototypes(DataSet.Images, EpisodeLoader.LabelMap)
    ProtoNet.SaveNetwork(Controls.OutputPath)

    #ValidationFeatures, ValidationLabels = EpisodeLoader.GetValidationSet()
    #FinalValidationAccuracy = ProtoNet.EvaluateAccuracy(ValidationFeatures, ValidationLabels)
    #Message = f"Final validation accuracy (final prototypes, {len(ValidationLabels)} held back training images): {100*FinalValidationAccuracy:.3f} %"
    #print(Message)
    #LogFile.W(f"\n{Message}")

    TestHandler = TEST.TestHandler(DataSet, EpisodeLoader, ProtoNet)
    TEST.NetworkTestMain(LogFile, Controls, ProtoNet, TestHandler)

    Duration = time.time() - StartTime
    Hours = int(Duration // 3600)
    Minutes = int((Duration % 3600) // 60)
    Seconds = int(Duration % 60)
    print(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}")
    LogFile.W(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}", NewLine=False)
    LogFile.Close()


if __name__ == "__main__":
    Main()