import os
import h5py
import numpy as np
import torch

import numpy as np
import matplotlib.pyplot as plt

class NoteFile():
    def __init__(self, FilePath: str):
        self.FilePath = FilePath
        os.makedirs(os.path.dirname(FilePath), exist_ok=True)
        self.File = open(FilePath, "w")

    def W(self, Text, NewLine: bool = True):
        self.File.write(str(Text) + "\n" if NewLine else str(Text))
        self.File.flush()

    def Close(self):
        self.File.close()

    def __del__(self):
        if not self.File.closed:
            self.File.close()

def LogDict(Log, Dictionary: dict, Indent: int = 0):
    for Key, Value in Dictionary.items():
        Prefix = "  " * Indent
        if isinstance(Value, dict):
            Log.W(f"{Prefix}{Key}:")
            LogDict(Log, Value, Indent + 1)
        elif isinstance(Value, torch.Tensor) and len(Value) > 0:
            Log.W(f"{Prefix}{Key}: Tensor {list(Value.shape)}")
        elif isinstance(Value, (list, np.ndarray)) and len(Value) > 0:
            Log.W(f"{Prefix}{Key} ({len(Value)} entries): {str(Value[:3]).rstrip(']')} ...]")
        else:
            Log.W(f"{Prefix}{Key}: {str(Value)[:50]}")

def LogHdf5Write(Log, Controls, HdfFile):
    Log.W(f"\nHDF5 File created as Outputs/{Controls.DataSet}_DINOv2Features.h5")
    Log.W(f"Meta data:")
    for Key, Value in HdfFile["MetaData"].attrs.items():
        Log.W(f"    {Key}: {Value}")
    def PrintLeaf(Name, Object):
        if isinstance(Object, h5py.Dataset):
            Log.W(f"    {Name}")
    Log.W(f"HDF5 paths:")
    HdfFile.visititems(PrintLeaf)

def ApplyDinoNetwork_FirstInstance(Log, MaxBatchSize, BatchSize, NumberOfBatches, DetailedLog, Images):
    Log.W(f"Maximum Batch Size: {MaxBatchSize}, Calculated Batch Size: {BatchSize}, Resulting Number of Batches: {NumberOfBatches}")
        
    if DetailedLog:
        for Index, _ in enumerate(Images["Images"]):
            Log.W(f"    {Index:>5} - {Images["Names"][Index]}: {Images["Images"][Index]}")
    
        Log.W("    Batching:")

def ApplyDinoNetwork_SecondInstance(Log, DetailedLog, BatchedImages, BatchFeatures, BatchedNames):
    if DetailedLog:
        for Index, _ in enumerate(BatchedImages):
            Vector = BatchFeatures[Index]
            Log.W(f"        {Index:>5} {BatchedNames[Index]:<30} [1 x {len(Vector):>5}] [{", ".join(f"{Entry: 7.4f}" for Entry in Vector[:5])}, ..., {", ".join(f"{Entry: 7.4f}" for Entry in Vector[-5:])}]")

def ApplyDinoNetwork_ThirdInstance(DetailedLog, Log, FeatureTensor, Images):
    if DetailedLog:
        Log.W("    Extracted Features:")
        Log.W(f"    Index   Name                           Feature Vector                                                                                  PIL Image")
        for Index in range(FeatureTensor.shape[0]):
            Vector = FeatureTensor[Index]
            Log.W(f"    {Index:>5} - {Images["Names"][Index]}: [{", ".join(f"{Entry: 7.4f}" for Entry in Vector[:5])}, ..., {", ".join(f"{Entry: 7.4f}" for Entry in Vector[-5:])}] {Images["Images"][Index]}")

def GenerateEncoder(Log, FeatureDimension, EmbeddingDimension, NumberOfHiddenLayers, Gamma, Indices, Dimensions):
    Log.W(f"Reducing from {FeatureDimension} to {EmbeddingDimension} Dimensions with {NumberOfHiddenLayers} hidden layers - Gamma = {Gamma}")
    Log.W("Indices   : [", NewLine=False)
    for Index in Indices:
        Log.W(f"{Index:>4}", NewLine=False)
    Log.W("]\nDimensions: [", NewLine=False)
    for Dimension in Dimensions:
        Log.W(f"{Dimension:>4}", NewLine=False)
    Log.W("]")

def Training(Controls, EpisodeLoader, Log):
    Log.W("\nStart main training loop\n==========================")
    Log.W("*----------------------------------------------------------*")
    Log.W("| Parameter                           | Variable | Value   |")
    Log.W("|                                     |          | (Input) |")
    Log.W("|-------------------------------------|----------|---------|")
    Log.W(f"| Number of Episodes                  |          | {Controls.NumberOfEpisodes:<4}    |")
    Log.W(f"| Number of Classes per Episode       | N_C      | {Controls.NumberOfClassesPerEpisode:<4}    |")
    Log.W(f"| Number of Support Samples per Class | N_S      | {Controls.NumberOfSupportSamplesPerClass:<4}    |")
    Log.W(f"| Number of Query Samples per Class   | N_Q      | {Controls.NumberOfQuerySamplesPerClass:<4}    |")
    Log.W("*----------------------------------------------------------*")

def WritelabelMap(Log, LabelMap):
    Log.W("\n*------------------------------*")
    Log.W("| Class                | Label |")
    Log.W("|----------------------|-------|")
    for Class in LabelMap.keys():
        Log.W(f"| {Class:<20} | {LabelMap[Class]:<5} |")
    Log.W("*------------------------------*")

def TrainingEpisode(Log, Controls, SelectedClasses, SupportIndices, QueryIndices, ValidationIndices, Episode):
    Log.W(f"  Selected classes   : {SelectedClasses}")
    Log.W(f"  Support indices    : {SupportIndices}")
    Log.W(f"  Query indices      : {QueryIndices}")
    Log.W(f"  Validation indices : {ValidationIndices}")
    
    Log.W(f"\n  Tensor shapes:")
    Log.W(f"  Support tensor   : [{Episode["SupportTensor"].size(0):>3}, {Episode["SupportTensor"].size(1):>3}, {Episode["SupportTensor"].size(2):>4}]")
    Log.W(f"  Query tensor     : [{Episode["QueryTensor"].size(0):>3}, {Episode["QueryTensor"].size(1):>3}, {Episode["QueryTensor"].size(2):>4}]")
    Log.W(f"  Validation tensor: [{Episode["ValidationTensor"].size(0):>3}, {Episode["ValidationTensor"].size(1):>3}, {Episode["ValidationTensor"].size(2):>4}]")

def PrintTrainingProcess(Log, Controls, RunningLoss, RunningAccuracy, RunningValidationAccuracy):
    NumberOfEpisodes = len(RunningAccuracy)
    Episodes = [Episode for Episode in range(NumberOfEpisodes)]

    Log.W("\n*---------------------------------------------*")
    Log.W("| Episode | Loss    | Accuracy   | Validation |")
    Log.W("|         |         |            | Accuracy   |")
    Log.W("|---------|---------|------------|------------|")
    for Index in range(NumberOfEpisodes):
        Log.W(f"| {Index+1:<7} | {round(RunningLoss[Index],5):<7} | {round(100*RunningAccuracy[Index],7):<10} | {round(100*RunningValidationAccuracy[Index],7):<10} |")
    Log.W("*---------------------------------------------*")
    StandardDeviationCutOff = int(NumberOfEpisodes/10)
    if StandardDeviationCutOff > 0:
        LossStandardDeviation   = np.std(RunningLoss[-StandardDeviationCutOff:])
        ValAccStandardDeviation = np.std([100*Entry for Entry in RunningValidationAccuracy[-StandardDeviationCutOff:]])
        Log.W(f"Standard deviation during the last 10% of training episodes")
        Log.W(f"Loss               : {round(LossStandardDeviation,3)}")
        Log.W(f"Validation Accuracy: {round(ValAccStandardDeviation,3)} %")

    with open(f"{Controls.OutputPath}TrainingProcess.csv", "w") as TrainingProcessCsv:
        TrainingProcessCsv.write("Episode Index,         Loss, Validation Accuracy\n")
        for Index, _ in enumerate(RunningAccuracy):
            TrainingProcessCsv.write(f"{Index+1:>13}, {round(RunningLoss[Index],10):12.8f}, {round(100*RunningAccuracy[Index],17):19.15f}\n")

    # Define plot
    Figure, LossAxis = plt.subplots(figsize=(12,5))
    AccuracyAxis = LossAxis.twinx()
    plt.title(f"Training info")

    # Set up X-Axis
    LossAxis.set_xlabel("Episode")
    plt.xlim(0, NumberOfEpisodes)

    # Y-Axis 1 (Loss)
    LossAxis.set_ylabel("Loss")
    LossAxis.set_ylim(0, 1.1 * np.amax(RunningLoss))
    LossAxis.plot(Episodes, RunningLoss, color="red", label="Loss")

    # Y-Axis 2 (Accuracy)
    AccuracyAxis.set_ylabel("Accuracy [%]")
    AccuracyAxis.set_ylim(0, 101)
    AccuracyAxis.plot(Episodes, [100 * Entry for Entry in RunningValidationAccuracy], color="navy"          , label="Validation Accuracy")
    AccuracyAxis.plot(Episodes, [100 * Entry for Entry in RunningAccuracy]          , color="cornflowerblue", label="Training Accuracy")
    
    Lines1, Labels1 = LossAxis.get_legend_handles_labels()
    Lines2, Labels2 = AccuracyAxis.get_legend_handles_labels()
    LossAxis.legend(Lines1 + Lines2, Labels1 + Labels2, loc="upper left")
    LossAxis.grid(True)
    plt.tight_layout()
    LossAxis.ticklabel_format(useOffset=False, style="plain")
    plt.savefig(f"{Controls.OutputPath}/TrainingInfo.png")

def DataExtraction_One(Log, Name, FeatureTensor, Label, LabelIndex, PositionIndex):
    Log.W(f"{PositionIndex:>5} ", NewLine=False)
    Log.W(f"{Name:<50} ", NewLine=False)
    Log.W(f"{Label:<20} ", NewLine=False)
    Log.W(f"{LabelIndex:<5} ", NewLine=False)
    FeatureTensorFront = FeatureTensor[:5]
    FeatureTensorBack  = FeatureTensor[-5:-1]
    Log.W("[ ", NewLine=False)
    for Entry in FeatureTensorFront:
        Log.W(f"{round(Entry.item(), 3):>6}, ", NewLine=False)
    Log.W("..., ", NewLine=False)
    for Entry in FeatureTensorBack:
        Log.W(f"{round(Entry.item(), 3):>6}, ", NewLine=False)
    Log.W(f"{round(FeatureTensor[-1].item(), 3):>6} ]")

def DataExtraction_Two(Log, PreparedData, Samples, Classes):
    Log.W(f"\nFinished Tensor of shape [{PreparedData["FeatureTensor"].shape[0]}, {PreparedData["FeatureTensor"].shape[1]}, {PreparedData["FeatureTensor"].shape[2]}]")
    Log.W("Extracted data:\nIndex Name                                               Label                Index Tensor")
    RunningIndex = 0
    for ClassIndex in range(PreparedData["FeatureTensor"].shape[0]):
        for EntryIndex in range(PreparedData["FeatureTensor"].shape[1]):
            Log.W(f"{RunningIndex:>5} ", NewLine=False)
            Log.W(f"{PreparedData["Names"][RunningIndex]:<50} ", NewLine=False)
            Log.W(f"{PreparedData["Labels"][RunningIndex]:<20} ", NewLine=False)
            Log.W(f"{PreparedData["Indices"][RunningIndex]:<5} ", NewLine=False)
            FeatureTensor = PreparedData["FeatureTensor"][ClassIndex][EntryIndex]
            FeatureTensorFront = FeatureTensor[:5]
            FeatureTensorBack  = FeatureTensor[-5:-1]
            Log.W("[ ", NewLine=False)
            for Entry in FeatureTensorFront:
                Log.W(f"{round(Entry.item(), 3):>6}, ", NewLine=False)
            Log.W("..., ", NewLine=False)
            for Entry in FeatureTensorBack:
                Log.W(f"{round(Entry.item(), 3):>6}, ", NewLine=False)
            Log.W(f"{round(FeatureTensor[-1].item(), 3):>6} ]")
            RunningIndex += 1