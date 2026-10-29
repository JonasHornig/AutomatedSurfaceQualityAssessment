import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from types import SimpleNamespace

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

def NameWidth(Names, Minimum: int = 20) -> int:
    # Column width for class names in the log tables: the longest name, but at least Minimum characters
    return max([Minimum] + [len(str(Name)) for Name in Names])

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

def TrainingSettings(Log, Controls, EpisodeLoader):
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
    Log.W(f"| Class Grouping                      |          | {Controls.ClassGrouping:<10} |")
    Log.W("*-------------------------------------------------------------*")

    Log.W("\nLabel map")
    W = NameWidth(EpisodeLoader.LabelMap.keys())
    Log.W(f"*{"-" * (W + 10)}*")
    Log.W(f"| {"Class":<{W}} | Label |")
    Log.W(f"|{"-" * (W + 2)}|-------|")
    for Class in EpisodeLoader.LabelMap.keys():
        Log.W(f"| {Class:<{W}} | {EpisodeLoader.LabelMap[Class]:<5} |")
    Log.W(f"*{"-" * (W + 10)}*")

    Log.W("\nTraining / validation split (the validation images are never used as support or query samples)")
    W = NameWidth(EpisodeLoader.CombinedClasses)
    Log.W(f"*{"-" * (W + 29)}*")
    Log.W(f"| {"Class":<{W}} | Training   | Validation  |")
    Log.W(f"|{"-" * (W + 2)}|------------|-------------|")
    for Class in EpisodeLoader.CombinedClasses:
        Log.W(f"| {Class:<{W}} | {len(EpisodeLoader.TrainingIndices[Class]):<10} | {len(EpisodeLoader.ValidationIndices[Class]):<11} |")
    Log.W(f"*{"-" * (W + 29)}*")
    if Controls.WriteDetailedDebugInfo:
        Log.W("Validation images (index within the class : feature vector):")
        for Class in EpisodeLoader.CombinedClasses:
            Sample, ClassName = Class.split("_", 1)
            Features = EpisodeLoader.Images[Sample][ClassName]["Training"]["DinoFeatures"]
            Log.W(f"{Class}")
            for Index in sorted(EpisodeLoader.ValidationIndices[Class].tolist()):
                Log.W(f"{Index:<5} : {FormatVector(Features[Index])}")

def TrainingEpisode(Log, EpisodeImages, Episode):
    # Support and query images of each selected class, in episode order. The validation images are fixed and listed in the training / validation split.
    Log.W("\nEpisode details")
    for ClassIndex, (Class, (SupportImages, QueryImages)) in enumerate(EpisodeImages.items()):
        for Kind, Images, Tensor in (("support", SupportImages, Episode["SupportTensor"]), ("query", QueryImages, Episode["QueryTensor"])):
            Log.W(f"{Class} (episode label {ClassIndex}), {Kind} images:")
            for TileIndex, Image in enumerate(Images):
                Log.W(f"{str(Image):<5}: {FormatVector(Tensor[ClassIndex][TileIndex])}")

def PrintTrainingProcess(Log, Controls, RunningLoss, RunningAccuracy, RunningValidationAccuracy):
    NumberOfEpisodes = len(RunningAccuracy)
    Episodes = list(range(1, NumberOfEpisodes + 1))

    if Controls.WriteDetailedDebugInfo:
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
        Log.W(f"\nStatistics over the last 10% of training episodes ({StandardDeviationCutOff} episodes)")
        Log.W(f"Loss mean                              : {LossMean:7.3f}")
        Log.W(f"Loss standard deviation                : {LossStandardDeviation:7.3f}")
        Log.W(f"Validation accuracy mean               : {ValAccMean:7.3f} %")
        Log.W(f"Validation accuracy standard deviation : {ValAccStandardDeviation:7.3f} %")

    print(f"Last episode: training accuracy = {round(100*RunningAccuracy[-1], 4)} %, validation accuracy = {round(100*RunningValidationAccuracy[-1], 4)} %")

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
    W = NameWidth(PreparedData["Labels"])
    for Label in dict.fromkeys(PreparedData["Labels"]):
        Log.W(f"    {Label:<{W}} {PreparedData["Labels"].count(Label):>6}")
    Log.W(f"Extracted data:\nIndex {"Name":<50} {"Label":<{W}} Index Tensor")
    for Index in range(PreparedData["FeatureTensor"].shape[0]):
        Log.W(f"{Index:>5} {PreparedData["Names"][Index]:<50} {PreparedData["Labels"][Index]:<{W}} {PreparedData["LabelIndices"][Index]:<5} {FormatVector(PreparedData["FeatureTensor"][Index])}")

def TestApplicationResults(Log, Controls, TestHandler):
    '''
    Results on the test set: every image (only with WriteDetailedDebugInfo), the per class accuracy and the confusion matrix, also saved as ConfusionMatrix.csv.
    Also reports the accuracy of telling the surface states apart (regardless of the material), the accuracy of telling
    the materials apart (only if the classes are <Material>_<SurfaceState>), and the surface state confidence.
    '''
    Data, LookUpTable = TestHandler.PreparedData, TestHandler.LookUpTable
    ConfusionMatrix   = Data["ConfusionMatrix"]
    Log.W("\nTest application of the proto net\n===================================")
    if Controls.WriteDetailedDebugInfo:
        W      = NameWidth(LookUpTable.values())
        Border = f"*{"-" * (2 * W + 66)}*"
        Log.W(Border)
        Log.W(f"| Index   | {"Correct Class":<{W + 6}} | {"Classification":<{W + 6}} | Confidence [%] | State confidence [%] |")
        Log.W(f"|---------|{"-" * (W + 8)}|{"-" * (W + 8)}|----------------|----------------------|")
        for Index in range(len(Data["Labels"])):
            Log.W(f"| {Index:<7} | {Data["LabelIndices"][Index]:>3} - {Data["Labels"][Index]:<{W}} | {Data["Classifications"][Index]:>3} - {LookUpTable[Data["Classifications"][Index]]:<{W}} | {100*Data["Probabilities"][Index]:>14.3f} | {100*Data["StateProbabilities"][Index]:>20.3f} |")
        Log.W(Border)
    CorrectClassifications = [A == B for A, B in zip(Data["LabelIndices"], Data["Classifications"])]
    Accuracy = 100*sum(CorrectClassifications) / len(CorrectClassifications)
    Log.W(f"Accuracy = {Accuracy:.3f} %")

    ClassNames = [LookUpTable[Index] for Index in range(len(LookUpTable))]
    Total      = ConfusionMatrix.sum()

    Log.W("\nTest results per class\n========================")
    Log.W("Confusion matrix (rows: true class, columns: predicted class)")
    W = NameWidth(ClassNames, Minimum=17)
    Log.W(f"{"":<{W + 3}} " + " ".join(f"{Index:>6}" for Index in range(len(ClassNames))) + " |  Images | Accuracy [%]")
    for Row, ClassName in enumerate(ClassNames):
        Images = ConfusionMatrix[Row].sum()
        Log.W(f"{Row:>2} {ClassName:<{W}} " + " ".join(f"{Count:>6}" for Count in ConfusionMatrix[Row]) + f" | {Images:>7} | " + (f"{100*ConfusionMatrix[Row, Row]/Images:>12.3f}" if Images else f"{'-':>12}")) # added prototypes have no test images

    CorrectState  = [True_ == Predicted for True_, Predicted in zip(Data["TrueStates"], Data["PredictedStates"])]
    StateAccuracy = 100 * sum(CorrectState) / len(CorrectState)
    if all("_" in ClassName for ClassName in ClassNames):
        Material = lambda Name: Name.split("_", 1)[0]
        MaterialAccuracy = f"{100 * sum(ConfusionMatrix[True_, Predicted] for True_ in range(len(ClassNames)) for Predicted in range(len(ClassNames)) if Material(ClassNames[True_]) == Material(ClassNames[Predicted])) / Total:.3f} %"
    else:
        MaterialAccuracy = "not applicable"
    Message = (f"Accuracy: all classes {100*np.trace(ConfusionMatrix)/Total:.3f} %, "
               f"surface state {StateAccuracy:.3f} %, material {MaterialAccuracy}")
    Log.W(Message)
    print(Message)

    # How confident the network is about the surface state, separately for correct and wrong surface states
    Confidences = np.array(Data["StateProbabilities"]) * 100
    Correct     = np.array(CorrectState)
    Log.W(f"Surface state confidence: mean {Confidences[Correct].mean():.2f} % for correct, "
          + (f"{Confidences[~Correct].mean():.2f} % for wrong surface states ({(~Correct).sum()} images)" if (~Correct).any() else "no wrong surface states"))

    with open(f"{Controls.OutputPath}/ConfusionMatrix.csv", "w") as ConfusionMatrixCsv:
        ConfusionMatrixCsv.write("TrueClass," + ",".join(ClassNames) + "\n")
        for Row, ClassName in enumerate(ClassNames):
            ConfusionMatrixCsv.write(f"{ClassName}," + ",".join(str(Count) for Count in ConfusionMatrix[Row]) + "\n")

def Darken(HexColour: str, Factor: float = 0.7) -> str:
    # Same hue, lower lightness: every RGB channel is scaled by Factor
    Red, Green, Blue = (int(HexColour[Index:Index + 2], 16) for Index in (1, 3, 5))
    return "#" + "".join(f"{int(Channel * Factor):02x}" for Channel in (Red, Green, Blue))

def Lighten(HexColour: str, Factor: float = 0.45) -> str:
    # Same hue, higher lightness: every RGB channel moves the fraction Factor towards white
    Red, Green, Blue = (int(HexColour[Index:Index + 2], 16) for Index in (1, 3, 5))
    return "#" + "".join(f"{int(Channel + (255 - Channel) * Factor):02x}" for Channel in (Red, Green, Blue))

def MaterialOf(ClassName: str) -> str:
    # Class names are <Material>_<State>, <Material>_<Reference>_<State> (added prototypes) or <State> (pooled prototypes, ClassGrouping = State)
    return ClassName.split("_")[0] if "_" in ClassName else ""

def ClassStyle(Samples: list[str], Classes: list[str], Names: list[str] = []):
    '''
    Colour and marker of any class name, shared by all PCA plots (static and interactive):
      colour = surface state (polished blue, scratched orange), so the two states are easy to tell apart
      marker = material (circle, square, diamond for the training materials, further markers for added materials)
      shade  = reference, if a material has several references with the same state (added prototypes, e.g. 1200 and 600 grit)
    Names fixes the order in which the markers and shades are assigned, so all plots of a run use the same ones.
    Returns a function Name -> (colour, marker), with the dictionaries MaterialMarker and StateColour as attributes (for legends).
    '''
    # Colour blind safe, assigned in the order of the classes (Polished, Scratched)
    StateColour    = dict(zip(Classes, ["#2a78d6", "#eb6834", "#1baf7a", "#7b3fa0"]))
    Markers        = ["o", "s", "D", "^", "v", "p", "<", ">"]
    MaterialMarker = dict(zip(Samples, Markers))
    Shades         = [lambda Colour: Colour, Darken, Lighten, lambda Colour: Darken(Colour, 0.45)]
    References     = {}
    def Style(Name: str) -> tuple[str, str]:
        Material, State = MaterialOf(Name), Name.split("_")[-1]
        if Material and Material not in MaterialMarker:
            MaterialMarker[Material] = Markers[len(MaterialMarker) % len(Markers)]
        Known = References.setdefault((Material, State), [])
        if Name not in Known:
            Known.append(Name)
        return Shades[Known.index(Name) % len(Shades)](StateColour.get(State, "#8a8a8a")), MaterialMarker.get(Material, "*")
    for Name in [f"{Sample}_{Class}" for Sample in Samples for Class in Classes] + Names:
        Style(Name)
    Style.MaterialMarker, Style.StateColour = MaterialMarker, StateColour
    return Style

def PlotValueDistribution(Values: dict, Threshold: float, Path: str, XLabel: str, LogScale: bool = False):
    '''
    Histogram of a score per tile, one panel per material, with the threshold from which a tile is scratched (score >= threshold).
    Values: {(material, state): array}. Used for the surface state score of the network.
    '''
    Materials = list(dict.fromkeys(Material for Material, _ in Values))
    Figure, Axes = plt.subplots(len(Materials), 1, figsize=(12, 2.6 * len(Materials)), sharex=True, squeeze=False)
    Bins = np.linspace(min(Value.min() for Value in Values.values()), max(Value.max() for Value in Values.values()) + 1, 60)
    for Axis, Material in zip(Axes[:, 0], Materials):
        for State, Colour in (("Polished", "#2a78d6"), ("Scratched", "#eb6834")):
            if (Material, State) in Values:
                Value = Values[(Material, State)]
                Wrong = (Value >= Threshold).sum() if State == "Polished" else (Value < Threshold).sum()
                Axis.hist(Value, bins=Bins, color=Colour, alpha=0.6, label=f"{State} ({len(Value)}, {Wrong} on the wrong side)")
        Axis.axvline(Threshold, color="#1f1f1f", linestyle="--", linewidth=1.2, label=f"Threshold {Threshold:g}")
        Axis.set_title(Material, fontsize=11); Axis.set_ylabel("Tiles")
        Axis.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0)) # outside, so it never covers the threshold
        if LogScale:
            Axis.set_yscale("log")
    Axes[-1, 0].set_xlabel(XLabel)
    Figure.tight_layout()
    Figure.savefig(Path, dpi=150, bbox_inches="tight")
    plt.close(Figure)

def MisclassifiedMasks(Data) -> tuple[np.ndarray, np.ndarray]:
    # Test images with a wrong surface state, and test images with the right surface state but the wrong class (i.e. the wrong material)
    WrongState = np.array([True_ != Predicted for True_, Predicted in zip(Data["TrueStates"], Data["PredictedStates"])])
    WrongClass = np.array([True_ != Predicted for True_, Predicted in zip(Data["LabelIndices"], Data["Classifications"])])
    return WrongState, WrongClass & ~WrongState

def PlotPrincipalComponentAnalysis(Controls, TestHandler, Data, ExplainedVarianceRatio, Name, Prototypes = [], Title:str = "Principal Component Analysis"):
    '''
    Colour encodes the surface state, the marker shape the material, see ClassStyle.
    Prototypes use the colour and shape of their class, drawn larger and with a red outline.
    Misclassified test images are outlined: black for a wrong surface state, grey for a wrong material only.
    '''
    LookUp          = TestHandler.LookUpTable
    Style           = ClassStyle(TestHandler.Samples, TestHandler.Classes, [LookUp[Index] for Index in range(len(LookUp))])
    TextColour      = "#1f1f1f"
    GridColour      = "#e3e3e3"
    WrongStateEdge  = "#000000"
    PrototypeEdge   = "#d00000"
    WrongClassEdge  = "#7a7a7a"

    Figure, Axis = plt.subplots(figsize=(12, 8))
    Axis.set_title(Title, fontsize=18, color=TextColour)
    Axis.set_xlabel(f"PC1 ({100*ExplainedVarianceRatio[0]:.2f}% of original variance)", fontsize=14, color=TextColour)
    Axis.set_ylabel(f"PC2 ({100*ExplainedVarianceRatio[1]:.2f}% of original variance)", fontsize=14, color=TextColour)
    Axis.grid(True, color=GridColour, linewidth=0.8)
    Axis.set_axisbelow(True)
    for Spine in Axis.spines.values():
        Spine.set_color("#b0b0b0")
    Axis.tick_params(colors="#555555")

    Members = np.array(TestHandler.PreparedData["Members"])
    WrongState, WrongClassOnly = MisclassifiedMasks(TestHandler.PreparedData)
    for Sample in TestHandler.Samples:
        for Class in TestHandler.Classes:
            Mask  = Members == f"{Sample}_{Class}"
            Marks = dict(zip(("c", "marker"), Style(f"{Sample}_{Class}")))
            Axis.scatter(Data[Mask, 0], Data[Mask, 1], s=25, alpha=0.5, zorder=2, **Marks)
            # Misclassified images are drawn again on top, so they are not hidden under the correct ones
            Axis.scatter(Data[Mask & WrongClassOnly, 0], Data[Mask & WrongClassOnly, 1], s=50, edgecolors=WrongClassEdge, linewidths=1.0, zorder=3, **Marks)
            Axis.scatter(Data[Mask & WrongState, 0], Data[Mask & WrongState, 1], s=50, edgecolors=WrongStateEdge, linewidths=1.0, zorder=4, **Marks)

    # Prototypes are ordered by the label map, so the look up table gives the class of each prototype.
    # Prototypes of pooled materials (ClassGrouping = State) are drawn as stars.
    for Index, Prototype in enumerate(Prototypes):
        Colour, Marker = Style(LookUp[Index])
        Axis.scatter(Prototype[0], Prototype[1], s=160, c=Colour, marker=Marker, edgecolors=PrototypeEdge, linewidths=1.6, zorder=5)

    # Separate legends per encoding, placed outside of the data area
    MaterialHandles = [Line2D([], [], marker=Style.MaterialMarker[Sample], linestyle="", markersize=10, markerfacecolor="#8a8a8a", markeredgecolor="white", label=Sample)
                       for Sample in TestHandler.Samples]
    StateHandles    = [Line2D([], [], marker="s", linestyle="", markersize=11, markerfacecolor=Style.StateColour[Class], markeredgecolor="white", label=Class)
                       for Class in TestHandler.Classes]
    StateHandles.append(Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(160), markerfacecolor="#8a8a8a", markeredgecolor=PrototypeEdge, markeredgewidth=1.6, label="Class prototype"))
    ErrorHandles    = [Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(50), markerfacecolor="#c8c8c8", markeredgecolor=WrongStateEdge, markeredgewidth=1.0,
                              label=f"Wrong surface state ({WrongState.sum()})"),
                       Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(50), markerfacecolor="#c8c8c8", markeredgecolor=WrongClassEdge, markeredgewidth=1.0,
                              label=f"Wrong material only ({WrongClassOnly.sum()})")]
    MaterialLegend = Axis.legend(handles=MaterialHandles, title="Material", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour)
    Axis.add_artist(MaterialLegend)
    StateLegend = Axis.legend(handles=StateHandles, title="Surface state", loc="upper left", bbox_to_anchor=(1.02, 0.66), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour, labelspacing=1.0)
    Axis.add_artist(StateLegend)
    ErrorLegend = Axis.legend(handles=ErrorHandles, title="Misclassified test images", loc="upper left", bbox_to_anchor=(1.02, 0.30), frameon=False, fontsize=12, title_fontsize=13, labelcolor=TextColour, labelspacing=1.0)

    Figure.tight_layout()
    Figure.savefig(f"{Controls.OutputPath}/{Name}.png", dpi=300, bbox_inches="tight", bbox_extra_artists=[MaterialLegend, StateLegend, ErrorLegend]) # all legends count for the figure size
    plt.close(Figure)

def DataSetHeader(LogFile):
    LogFile.W("\nLoading data set\n==================")
    LogFile.W("*-------------------------------------------------------------------*")
    LogFile.W("| Class      | Material                   | Mode       | Number     |")
    LogFile.W("|            |                            |            | of images  |")
    LogFile.W("|------------|----------------------------|------------|------------|")
