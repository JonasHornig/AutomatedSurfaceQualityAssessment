import time
StartTime = time.time()

import torch
import os

from tqdm import tqdm
from dataclasses import dataclass, field

import Modules.DataSetHandling as DATA
import Modules.NeuralNetwork as NN
import Modules.Logging as LOG
import Modules.EpisodeHandler as EPH

import PreprocessImages as PPI

@dataclass
class ControlVariables:
    # Inputs
    Modes      : tuple = ("Training", "Testing")
    DataSet    : str   = "CompleteDataSet" #  --  DummyDataSet  --  SmallDataSet  --  CompleteDataSet

    FeatureDimension     : int   = 768
    EmbeddingDimension   : int   = 5
    NumberOfHiddenLayers : int   = 4
    Gamma                : float = 0.75

    NumberOfEpisodes               : int   = 50
    LearningRate                   : float = 1e-4
    NumberOfClassesPerEpisode      : int   = 5     # N_C =< K
    NumberOfSupportSamplesPerClass : int   = 9     # N_S
    NumberOfQuerySamplesPerClass   : int   = 1     # N_Q

    # Flags
    #  --  True  --  False  --
    WriteDetailedDebugInfo : bool = False
    
    # Parameter initialisation
    NumberOfClasses : int = 0 # K
    
    def __post_init__(self):
        BaseDirectory = os.path.dirname(os.path.abspath(__file__))
        self.DataSetPath = f"{BaseDirectory}/DataSets/{self.DataSet}"
        self.OutputPath : str   = f"{BaseDirectory}/Outputs/"

def Main():
    Controls  = ControlVariables()
    print("")
    LogFile = LOG.NoteFile(f"{Controls.OutputPath}/Training.log")
    LogFile.W("\n**************************\n*  Training a Proto Net  *\n**************************")
    DataSet = DATA.DataSet()
    DataSet.LoadDataSet(LogFile, Controls)

    if not DataSet.ProcessedDataSet:
        LogFile = LOG.NoteFile(f"Outputs/Preprocessing.log")
        PPI.Main(LogFile)

    ProtoNet  = NN.ProtoNet(LogFile, Controls)
    Optimizer = torch.optim.AdamW(ProtoNet.Encoder.parameters(), lr=Controls.LearningRate)
    Optimizer.zero_grad()
    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=Controls.NumberOfEpisodes)

    EpisodeLoader = EPH.EpisodeLoader(DataSet.Images, DataSet.GetParameters(), Controls)
    LOG.Training(Controls, EpisodeLoader, LogFile)
    EpisodeLoader.GenerateLabelMap(Controls.OutputPath)
    LOG.WritelabelMap(LogFile, EpisodeLoader.LabelMap)

    RunningLoss     = []
    RunningAccuracy = []
    for EpisodeIndex in tqdm(range(1, Controls.NumberOfEpisodes + 1), desc="Training", unit="ep"):
        if Controls.WriteDetailedDebugInfo:
            LogFile.W(f"\nEpisode {EpisodeIndex:>5} out of {Controls.NumberOfEpisodes:>5}")

        ActiveEpisode = EpisodeLoader.GetRandomEpisode(LogFile, Controls)
        Loss, LossInfo = ProtoNet.CalculateLoss(ActiveEpisode)

        RunningLoss.append(LossInfo["Loss"])
        RunningAccuracy.append(LossInfo["ValidationAccuracy"])

        Loss.backward()
        Optimizer.step()

    LOG.PrintTrainingProcess(LogFile, Controls, RunningLoss, RunningAccuracy)

    LogFile.W(f"\nTraining complete")
    Duration = time.time() - StartTime
    Hours = int(Duration // 3600)
    Minutes = int((Duration % 3600) // 60)
    Seconds = int(Duration % 60)
    print(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}")
    LogFile.W(f"Runtime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}", NewLine=False)
    LogFile.Close()


if __name__ == "__main__":
    Main()