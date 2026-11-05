import time
StartTime = time.time()

import argparse
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
    FeatureExtractor : str = "DINOv3"      #  --  "DINOv3" (ViT-B/16)  --  "DINOv3-H16plus" (ViT-H+/16)  --  "DINOv1" (legacy)

    # Network. The input dimension is taken from the feature files (768 for DINOv3 ViT-B/16)
    EmbeddingDimension   : int   = 8 # 2 to 32 performed equally well in a study (5 seeds each), 8 gave the most consistent results
    NumberOfHiddenLayers : int   = 2 # 1 to 4 performed equally well in a study (5 seeds each), 2 gave the most consistent results; 0 and 8 were worse
    Gamma                : float = 0.75

    # Training. Chosen in a study on the validation accuracy (5 seeds per setting, see README): the learning rate and the number of episodes
    # matter most, the number of support and query samples and the weight decay hardly matter
    NumberOfEpisodes               : int   = 2000  # With LearningRate 1e-3: 97.8 % validation accuracy (1e-4 with 5000 episodes: 97.5 %), confirmed with five further seeds
    LearningRate                   : float = 1e-3
    WeightDecay                    : float = 1e-2  # AdamW weight decay (1e-2 is the PyTorch default)
    UseLearningRateScheduler       : bool  = True  # Cosine annealing of the learning rate from LearningRate to 0 over NumberOfEpisodes. Makes the results more consistent between seeds
    Seed                           : int   = 333   # Seed for all random number generators, makes training runs reproducible
    NumberOfClassesPerEpisode      : int   = 6     # N_C =< K
    NumberOfSupportSamplesPerClass : int   = 5     # N_S
    NumberOfQuerySamplesPerClass   : int   = 10    # N_Q
    ValidationFraction             : float = 0.25  # Fraction of the training images of each class held back for validation, never used for support or query samples
    ClassGrouping                  : str   = "MaterialAndState" #  --  "MaterialAndState": one class (prototype) per material and surface state, e.g. 18201-GJSCastIron_Polished  --  "State": materials pooled, one class per surface state (N_C = 2)

    # Outputs
    RunName : str = "" # Name of the output folder Outputs/<RunName>/. Empty: date and time of the start of the run, so runs never overwrite each other

    InteractivePCA         : bool = True  # Also writes PCA_Interactive.html: the test set PCA to explore in a web browser, with image previews
    InteractivePCAImages   : int  = 1000  # Correctly classified test images shown in PCA_Interactive.html (all misclassified ones are shown). About 5 kB each

    # Flags
    #  --  True  --  False  --
    WriteDetailedDebugInfo : bool = False # True: also logs every episode, image and feature vector (large log files). Enabling this may severely impact performance
    GenerateRandomSeed     : bool = False # True: a random Seed is drawn for the run (saved in Settings.json, so the run can still be reproduced)

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
        # All settings of the run are saved next to its results, so every run can be reproduced
        with open(f"{self.OutputPath}/Settings.json", "w") as SettingsFile:
            json.dump(asdict(self), SettingsFile, indent=4)

    def SetSeed(self):
        # Episode sampling uses Python's random module and torch, the weight initialisation uses torch
        if self.GenerateRandomSeed:
            self.Seed = random.randint(100000000, 999999999)
        random.seed(self.Seed)
        np.random.seed(self.Seed)
        torch.manual_seed(self.Seed)

def Main():
    Parser = argparse.ArgumentParser(description="Train the prototypical network with the settings in ControlVariables.")
    Parser.add_argument("--run-name", default=ControlVariables.RunName, help="Name of the output folder src/Outputs/<RunName>/ (default: date and time)")
    Arguments = Parser.parse_args()
    if Arguments.run_name and os.path.exists(f"{os.path.dirname(os.path.abspath(__file__))}/Outputs/{Arguments.run_name}"):
        Parser.error(f"src/Outputs/{Arguments.run_name} already exists, choose another run name")
    Controls = ControlVariables(RunName=Arguments.run_name)
    LogFile  = LOG.NoteFile(f"{Controls.OutputPath}/Training.log")

    print(f"\nRun {Controls.RunName}\n=======================")
    LogFile.W("\n****************************\n*  Training the Proto Net  *\n****************************")
    LogFile.W(f"Run {Controls.RunName} - all settings are saved in Settings.json")

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

    # The final prototypes are computed from the training images only, so the held back validation images stay unseen
    ProtoNet.ComputePrototypes(EpisodeLoader.GetTrainingSet())
    ProtoNet.SaveNetwork(Controls.OutputPath)

    ValidationFeatures, ValidationLabels = EpisodeLoader.GetValidationSet()
    Classifications = ProtoNet.Classify(ValidationFeatures)
    ClassAccuracy   = (Classifications == ValidationLabels).float().mean().item()
    State           = lambda Label: TEST.SurfaceState(EpisodeLoader.LookUpTable[Label])
    StateAccuracy   = sum(State(Predicted) == State(True_) for Predicted, True_ in zip(Classifications.tolist(), ValidationLabels.tolist())) / len(ValidationLabels)
    Message = (f"Final validation accuracy (final prototypes, {len(ValidationLabels)} held back training images): "
               f"{100*ClassAccuracy:.3f} %, surface state {100*StateAccuracy:.3f} %")
    print(Message)
    LogFile.W(f"\n{Message}")

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