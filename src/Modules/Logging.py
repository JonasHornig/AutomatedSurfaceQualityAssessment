import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

class NoteFile():
    def __init__(self, FilePath: str):
        self.FilePath = FilePath
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
    Log.W(f"\nHDF5 File created as {HdfFile.filename}")
    Log.W(f"Meta data:")
    for Key, Value in HdfFile["MetaData"].attrs.items():
        Log.W(f"    {Key}: {Value}")
    def PrintLeaf(Name, Object):
        if isinstance(Object, h5py.Dataset):
            Log.W(f"    {Name}")
    Log.W(f"HDF5 paths:")
    HdfFile.visititems(PrintLeaf)

def FormatVector(Vector, NumberOfEntries: int = 5) -> str:
    # Shortened representation of a long vector: its first and last entries
    Front = ", ".join(f"{Entry: 7.4f}" for Entry in Vector[:NumberOfEntries].tolist())
    Back  = ", ".join(f"{Entry: 7.4f}" for Entry in Vector[-NumberOfEntries:].tolist())
    return f"[{Front}, ..., {Back}] ({len(Vector)} entries)"

def FeatureBatch(Log, BatchIndex, NumberOfBatches, Names, Features):
    Log.W(f"    Batch {BatchIndex+1} out of {NumberOfBatches}")
    for Index, Name in enumerate(Names):
        Log.W(f"    {Index:>5} {Name:<40} {FormatVector(Features[Index])}")

def GenerateEncoder(Log, FeatureDimension, EmbeddingDimension, NumberOfHiddenLayers, Gamma, Indices, Dimensions):
    Log.W(f"Reducing from {FeatureDimension} to {EmbeddingDimension} Dimensions with {NumberOfHiddenLayers} hidden layers - Gamma = {Gamma}")
    Log.W("Indices   : [", NewLine=False)
    for Index in Indices:
        Log.W(f"{Index:>4}", NewLine=False)
    Log.W("]\nDimensions: [", NewLine=False)
    for Dimension in Dimensions:
        Log.W(f"{Dimension:>4}", NewLine=False)
    Log.W("]")

def Training(Log, Controls):
    Log.W("\nStart main training loop\n==========================")
    Log.W("*-------------------------------------------------------------*")
    Log.W("| Parameter                           | Variable | Value      |")
    Log.W("|                                     |          | (Input)    |")
    Log.W("|-------------------------------------|----------|------------|")
    Log.W(f"| Seed                                |          | {Controls.Seed:<10} |")
    Log.W(f"| Number of Episodes                  |          | {Controls.NumberOfEpisodes:<10} |")
    Log.W(f"| Learning Rate                       |          | {Controls.LearningRate:<10.1e} |")
    Log.W(f"| Learning Rate Scheduler (cosine)    |          | {str(Controls.UseLearningRateScheduler):<10} |")
    Log.W(f"| Weight Decay                        |          | {Controls.WeightDecay:<10.1e} |")
    Log.W(f"| Number of Classes per Episode       | N_C      | {Controls.NumberOfClassesPerEpisode:<10} |")
    Log.W(f"| Number of Support Samples per Class | N_S      | {Controls.NumberOfSupportSamplesPerClass:<10} |")
    Log.W(f"| Number of Query Samples per Class   | N_Q      | {Controls.NumberOfQuerySamplesPerClass:<10} |")
    Log.W(f"| Validation Fraction                 |          | {Controls.ValidationFraction:<10} |")
    Log.W("*-------------------------------------------------------------*")

def ValidationSplit(Log, EpisodeLoader, DetailedLog: bool = False):
    Log.W("\nHeld back validation images (never used as support or query samples)")
    Log.W("*-------------------------------------------------*")
    Log.W("| Class                | Training   | Validation  |")
    Log.W("|----------------------|------------|-------------|")
    for Class in EpisodeLoader.CombinedClasses:
        Log.W(f"| {Class:<20} | {len(EpisodeLoader.TrainingIndices[Class]):<10} | {len(EpisodeLoader.ValidationIndices[Class]):<11} |")
    Log.W("*-------------------------------------------------*")
    if DetailedLog:
        Log.W("Validation image indices within each class:")
        for Class in EpisodeLoader.CombinedClasses:
            Log.W(f"  {Class:<20} : {sorted(EpisodeLoader.ValidationIndices[Class].tolist())}")

def WritelabelMap(Log, LabelMap):
    Log.W("\n*------------------------------*")
    Log.W("| Class                | Label |")
    Log.W("|----------------------|-------|")
    for Class in LabelMap.keys():
        Log.W(f"| {Class:<20} | {LabelMap[Class]:<5} |")
    Log.W("*------------------------------*")

def TrainingEpisode(Log, EpisodeIndices, Episode):
    # Image indices within each selected class, in episode order. The validation images are fixed and listed in the validation split.
    Log.W(f"  Class                | Support indices | Query indices")
    for Class, (SupportIndices, QueryIndices) in EpisodeIndices.items():
        Log.W(f"  {Class:<20} | {SupportIndices} | {QueryIndices}")

    Log.W(f"\n  Tensor shapes:")
    Log.W(f"  Support tensor   : [{Episode["SupportTensor"].size(0):>3}, {Episode["SupportTensor"].size(1):>3}, {Episode["SupportTensor"].size(2):>4}]")
    Log.W(f"  Query tensor     : [{Episode["QueryTensor"].size(0):>3}, {Episode["QueryTensor"].size(1):>3}, {Episode["QueryTensor"].size(2):>4}]")
    Log.W(f"  Validation tensor: [{Episode["ValidationTensor"].size(0):>8}, {Episode["ValidationTensor"].size(1):>4}]")

def PrintTrainingProcess(Log, Controls, RunningLoss, RunningAccuracy, RunningValidationAccuracy):
    NumberOfEpisodes = len(RunningAccuracy)
    Episodes = list(range(1, NumberOfEpisodes + 1))

    Log.W("\n*-------------------------------------------------*")
    Log.W("| Episode | Loss      | Training   | Validation   |")
    Log.W("|         |           | Accuracy % | Accuracy %   |")
    Log.W("|---------|-----------|------------|--------------|")
    for Index in range(NumberOfEpisodes):
        Log.W(f"| {Index+1:<7} | {RunningLoss[Index]:>9.6f} | {100*RunningAccuracy[Index]:>10.3f} | {100*RunningValidationAccuracy[Index]:>12.3f} |")
    Log.W("*-------------------------------------------------*")
    StandardDeviationCutOff = int(NumberOfEpisodes/10)
    if StandardDeviationCutOff > 0:
        LossMean                = np.mean(RunningLoss[-StandardDeviationCutOff:])
        LossStandardDeviation   = np.std(RunningLoss[-StandardDeviationCutOff:])
        ValAccMean              = np.mean([100*Entry for Entry in RunningValidationAccuracy[-StandardDeviationCutOff:]])
        ValAccStandardDeviation = np.std([100*Entry for Entry in RunningValidationAccuracy[-StandardDeviationCutOff:]])
        Log.W(f"Statistics over the last 10% of training episodes ({StandardDeviationCutOff} episodes)")
        Log.W(f"Loss mean                              : {LossMean:.3f}")
        Log.W(f"Loss standard deviation                : {LossStandardDeviation:.3f}")
        Log.W(f"Validation accuracy mean               : {ValAccMean:.3f} %")
        Log.W(f"Validation accuracy standard deviation : {ValAccStandardDeviation:.3f} %")

    print(f"Final accuracy: Trainin accuracy = {round(100*RunningAccuracy[-1], 4)} %; Validation Accuracy = {round(100*RunningValidationAccuracy[-1], 4)}")

    with open(f"{Controls.OutputPath}/TrainingProcess.csv", "w") as TrainingProcessCsv:
        # Accuracies in %. Training accuracy is measured on the query samples, validation accuracy on the held back validation images
        # Plain comma separated values without padding, so the file can be read directly by e.g. pandas or Excel
        TrainingProcessCsv.write("Episode,Loss,TrainingAccuracyPercent,ValidationAccuracyPercent\n")
        for Index, _ in enumerate(RunningAccuracy):
            TrainingProcessCsv.write(f"{Index+1},{RunningLoss[Index]:.6f},{100*RunningAccuracy[Index]:.3f},{100*RunningValidationAccuracy[Index]:.3f}\n")

    # Define plot
    Figure, LossAxis = plt.subplots(figsize=(12,5))
    AccuracyAxis = LossAxis.twinx()
    plt.title(f"Training info")

    # Set up X-Axis
    LossAxis.set_xlabel("Episode")
    plt.xlim(1, NumberOfEpisodes)

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
    LossAxis.legend(Lines1 + Lines2, Labels1 + Labels2, loc="center right")
    LossAxis.grid(True)
    plt.tight_layout()
    LossAxis.ticklabel_format(useOffset=False, style="plain")
    plt.savefig(f"{Controls.OutputPath}/TrainingInfo.png")
    plt.close(Figure)

def DataExtraction(Log, PreparedData):
    Log.W("\nExtracting data from the data set for the test application")
    Log.W(f"Finished Tensor of shape [{PreparedData["FeatureTensor"].shape[0]}, {PreparedData["FeatureTensor"].shape[1]}]")
    Log.W("Number of images per class:")
    for Label in dict.fromkeys(PreparedData["Labels"]):
        Log.W(f"    {Label:<20} {PreparedData["Labels"].count(Label):>6}")
    Log.W("Extracted data:\nIndex Name                                               Label                Index Tensor")
    for Index in range(PreparedData["FeatureTensor"].shape[0]):
        Log.W(f"{Index:>5} {PreparedData["Names"][Index]:<50} {PreparedData["Labels"][Index]:<20} {PreparedData["LabelIndices"][Index]:<5} {FormatVector(PreparedData["FeatureTensor"][Index])}")

def TestApplicationResults(Log, Controls, TestHandler):
    Log.W("\nTest application of the proto net\n===================================")
    Log.W("*------------------------------------------------------------------------------------*")
    Log.W("| Index   | Correct Class              | Classification             | Confidence [%] |")
    Log.W("|---------|----------------------------|----------------------------|----------------|")
    for Index in range(len(TestHandler.PreparedData["Labels"])):
        Log.W(f"| {Index:<7} | {TestHandler.PreparedData['LabelIndices'][Index]:>3} - {TestHandler.PreparedData['Labels'][Index]:<20} | {TestHandler.PreparedData['Classifications'][Index]:>3} - {TestHandler.LookUpTable[TestHandler.PreparedData['Classifications'][Index]]:<20} | {100*TestHandler.PreparedData['Probabilities'][Index]:>14.3f} |")
    Log.W("*------------------------------------------------------------------------------------*")
    CorrectClassifications = [A == B for A, B in zip(TestHandler.PreparedData["LabelIndices"], TestHandler.PreparedData["Classifications"])]
    Accuracy = 100*sum(CorrectClassifications) / len(CorrectClassifications)
    Log.W(f"Accuracy = {Accuracy:.3f} %")
    print(f"Total test accuracy = {round(Accuracy, 4)} %")

    ClassNames = [TestHandler.LookUpTable[Index] for Index in range(len(TestHandler.LookUpTable))]
    Total      = TestHandler.PreparedData["ConfusionMatrix"].sum()

    def GroupAccuracy(Part):
        # Fraction of test images whose predicted class has the same material (Part 0) or surface state (Part 1) as the true class
        return 100 * sum(TestHandler.PreparedData["ConfusionMatrix"][True_, Predicted] for True_ in range(len(ClassNames)) for Predicted in range(len(ClassNames))
                         if ClassNames[True_].split("_", 1)[Part] == ClassNames[Predicted].split("_", 1)[Part]) / Total

    Log.W("\nTest results per class\n========================")
    Log.W("Confusion matrix (rows: true class, columns: predicted class)")
    Log.W(f"{"":<20} " + " ".join(f"{Index:>6}" for Index in range(len(ClassNames))) + " |  Images | Accuracy [%]")
    for Row, ClassName in enumerate(ClassNames):
        Images = TestHandler.PreparedData["ConfusionMatrix"][Row].sum()
        Log.W(f"{Row:>2} {ClassName:<17} " + " ".join(f"{Count:>6}" for Count in TestHandler.PreparedData["ConfusionMatrix"][Row]) + f" | {Images:>7} | {100*TestHandler.PreparedData["ConfusionMatrix"][Row, Row]/Images:>12.3f}")
    Log.W(f"Accuracy: all classes {100*np.trace(TestHandler.PreparedData["ConfusionMatrix"])/Total:.3f} %, surface state {GroupAccuracy(1):.3f} %, material {GroupAccuracy(0):.3f} %")

    with open(f"{Controls.OutputPath}/ConfusionMatrix.csv", "w") as ConfusionMatrixCsv:
        ConfusionMatrixCsv.write("TrueClass," + ",".join(ClassNames) + "\n")
        for Row, ClassName in enumerate(ClassNames):
            ConfusionMatrixCsv.write(f"{ClassName}," + ",".join(str(Count) for Count in TestHandler.PreparedData["ConfusionMatrix"][Row]) + "\n")

def PlotPrincipalComponentAnalysis(Controls, TestHandler, Data, ExplainedVarianceRatio, Name, Prototypes = [], Title:str = "Principal Component Analysis"):
    '''
    Colour encodes the material (sample), the marker shape encodes the surface state (class).
    Prototypes use the colour and shape of their class, drawn larger and with a dark outline.
    '''
    # Colour blind safe categorical colours, validated as a set of three for scatter plots. Assigned in fixed sample order.
    MaterialColours = ["#2a78d6", "#eb6834", "#1baf7a"]
    StateMarkers    = {"Polished": "o", "Scratched": "^"}
    FallbackMarkers = ["s", "D", "v", "P"]
    TextColour      = "#1f1f1f"
    GridColour      = "#e3e3e3"
    
    MaterialMarkers = {"18201": "o", "22116": "^", "22348": "D"}
    StateColours    = {"Polished": "#ac1313", "Scratched": "#3ae90e"}
    
    if len(TestHandler.Samples) > len(MaterialMarkers):
        raise ValueError(f"Only {len(MaterialMarkers)} material markers available but got {len(TestHandler.Samples)} materials - extend MaterialMarkers in Logging.py")
    
    ClassColours  = {}
    for Class in TestHandler.Classes:
        ClassColours[Class] = StateColours.get(Class)
    SampleMarkers = {}
    for Sample in TestHandler.Samples:
        SampleMarkers[Sample] = MaterialMarkers.get(Sample)
    
    Figure, Axis = plt.subplots(figsize=(12, 8))
    Axis.set_title(Title, fontsize=18, color=TextColour)
    Axis.set_xlabel(f"PC1 ({100*ExplainedVarianceRatio[0]:.2f}% of original variance)", fontsize=14, color=TextColour)
    Axis.set_ylabel(f"PC2 ({100*ExplainedVarianceRatio[1]:.2f}% of original variance)", fontsize=14, color=TextColour)
    Axis.grid(True, color=GridColour, linewidth=0.8)
    Axis.set_axisbelow(True)
    for Spine in Axis.spines.values():
        Spine.set_color("#b0b0b0")
    Axis.tick_params(colors="#555555")

    # Examples
    Labels = np.array(TestHandler.PreparedData["Labels"])
    for Sample in TestHandler.Samples:
        for Class in TestHandler.Classes:
            Mask = Labels == f"{Sample}_{Class}"
            Axis.scatter(Data[Mask, 0], Data[Mask, 1], s=25, c=ClassColours[Class], marker=SampleMarkers[Sample], alpha=0.5, zorder=2)

    # Prototypes
    for Index, Prototype in enumerate(Prototypes):
        Sample, Class = TestHandler.LookUpTable[Index].split("_", 1)
        Axis.scatter(Prototype[0], Prototype[1], s=200, c=ClassColours[Class], marker=SampleMarkers[Sample], edgecolors=TextColour, linewidths=1, zorder=3)

    # Two separate legends, one per encoding, placed outside of the data area
    MaterialHandles = [Line2D([], [], marker=SampleMarkers[Sample], linestyle="", markersize=11, markerfacecolor="#8a8a8a", markeredgecolor="white", label=Sample) for Sample in TestHandler.Samples]
    StateHandles    = [Line2D([], [], marker="s", linestyle="", markersize=10, markerfacecolor=ClassColours[Class], markeredgecolor="white", label=Class) for Class in TestHandler.Classes]
    StateHandles.append(Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(200), markerfacecolor="#8a8a8a", markeredgecolor=TextColour, markeredgewidth=1, label="Class prototype"))
    MaterialLegend = Axis.legend(handles=MaterialHandles, title="Material", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour)
    Axis.add_artist(MaterialLegend)
    Axis.legend(handles=StateHandles, title="Surface state", loc="upper left", bbox_to_anchor=(1.02, 0.62), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour, labelspacing=1.0)

    '''
    if len(TestHandler.Samples) > len(MaterialColours):
        raise ValueError(f"Only {len(MaterialColours)} material colours available but got {len(TestHandler.Samples)} materials - extend MaterialColours in Logging.py")
    SampleColour = {Sample: MaterialColours[Index] for Index, Sample in enumerate(TestHandler.Samples)}
    ClassMarker  = {}
    for Class in TestHandler.Classes:
        ClassMarker[Class] = StateMarkers.get(Class, FallbackMarkers[len(ClassMarker) % len(FallbackMarkers)])

    Figure, Axis = plt.subplots(figsize=(12, 8))
    Axis.set_title(Title, fontsize=18, color=TextColour)
    Axis.set_xlabel(f"PC1 ({100*ExplainedVarianceRatio[0]:.2f}% of original variance)", fontsize=14, color=TextColour)
    Axis.set_ylabel(f"PC2 ({100*ExplainedVarianceRatio[1]:.2f}% of original variance)", fontsize=14, color=TextColour)
    Axis.grid(True, color=GridColour, linewidth=0.8)
    Axis.set_axisbelow(True)
    for Spine in Axis.spines.values():
        Spine.set_color("#b0b0b0")
    Axis.tick_params(colors="#555555")

    Labels = np.array(TestHandler.PreparedData["Labels"])
    for Sample in TestHandler.Samples:
        for Class in TestHandler.Classes:
            Mask = Labels == f"{Sample}_{Class}"
            Axis.scatter(Data[Mask, 0], Data[Mask, 1], s=25, c=SampleColour[Sample], marker=ClassMarker[Class], alpha=0.3, zorder=2)

    # Prototypes are ordered by the label map, so the look up table gives the class of each prototype
    for Index, Prototype in enumerate(Prototypes):
        Sample, Class = TestHandler.LookUpTable[Index].split("_", 1)
        Axis.scatter(Prototype[0], Prototype[1], s=200, c=SampleColour[Sample], marker=ClassMarker[Class], edgecolors=TextColour, linewidths=1, zorder=3)

    # Two separate legends, one per encoding, placed outside of the data area
    MaterialHandles = [Line2D([], [], marker="s", linestyle="", markersize=11, markerfacecolor=SampleColour[Sample], markeredgecolor="white", label=Sample) for Sample in TestHandler.Samples]
    StateHandles    = [Line2D([], [], marker=ClassMarker[Class], linestyle="", markersize=10, markerfacecolor="#8a8a8a", markeredgecolor="white", label=Class) for Class in TestHandler.Classes]
    StateHandles.append(Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(200), markerfacecolor="#8a8a8a", markeredgecolor=TextColour, markeredgewidth=1, label="Class prototype"))
    MaterialLegend = Axis.legend(handles=MaterialHandles, title="Material", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour)
    Axis.add_artist(MaterialLegend)
    Axis.legend(handles=StateHandles, title="Surface state", loc="upper left", bbox_to_anchor=(1.02, 0.62), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour, labelspacing=1.0)
    '''

    Figure.tight_layout()
    Figure.savefig(f"{Controls.OutputPath}/{Name}.png", dpi=300, bbox_inches="tight")
    plt.close(Figure)