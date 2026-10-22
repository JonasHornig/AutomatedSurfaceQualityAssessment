'''
Adds prototypes of a new material to a trained network without retraining it (few-shot, Snell et al. 2017).
This is optional: the trained prototypes often classify a new material well already. New prototypes are mainly worth it
for a material whose microstructure is clearly different from the training materials.

Reference images (RASI, 50x) in one folder per reference, without further folder levels:
    src/DataSets/AdditionalPrototypes/<Material>_<Reference>_<State>/   e.g. 22188-GJS400CastIron_1200Grit_Scratched
    src/DataSets/AdditionalPrototypes/<Material>_<State>/               e.g. 22188-GJS400CastIron_Polished
<Material> and <Reference> must not contain "_", <State> has to be a surface state of the network (Polished or Scratched).
Every folder becomes one prototype: the mean embedding of all tiles of its images. The encoder is not changed.

Steps:
  1. Check: does the network already classify the surface state of the reference tiles correctly? Then no prototypes are added
     (unless SkipExistingCheck / --skip-existing-check)
  2. Validation: every image is held out once (a folder with a single image: each half of it). Its tiles are classified with the
     existing prototypes and the new prototypes, built without the held out image
  3. If the validation passes (or IgnoreValidation / --ignore-validation): new network folder src/Outputs/<NetworkRun>+<Material>/ with the unchanged encoder, the existing and
     the new prototypes and the extended label map. It is used like a training run, e.g. ClassifyImage.py --network <NetworkRun>+<Material>
Every run also shows where the reference tiles and the new prototypes lie compared with the training materials: PCA_NewPrototypes.png
and the interactive PCA_NewPrototypes.html (with a preview of every tile). For these, the test set of the training data is classified with
the existing and the new prototypes (TestSet.log, ConfusionMatrix.csv), which also shows how many training test images the new prototypes take over.
The prototypes depend on the encoder: after retraining the network, they have to be added again from the same images.

Command line:
    python src/AddNewMaterialPrototype.py <Material> --network <training run in src/Outputs/> [--skip-existing-check] [--ignore-validation]
'''
import argparse
import json
import os
import shutil
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from types import SimpleNamespace
from PIL import Image
from sklearn.decomposition import PCA

import ClassifyImage as CI
import Modules.DataSetHandling as DATA
import Modules.InteractivePCA as IPCA
import Modules.Logging as LOG
import Modules.NeuralNetwork as NN
import Modules.NetworkTesting as TEST

@dataclass
class ControlVariables:
    Material                 : str   = ""    # Name of the new material, the start of its folder names in PrototypePath
    NetworkRun               : str   = ""    # Trained network in src/Outputs/ (may already contain added prototypes)
    PrototypePath            : str   = ""    # Folder with the reference image folders. Empty: src/DataSets/AdditionalPrototypes
    RequiredAccuracy_Percent : float = 95.0  # Share of tiles with the correct surface state required in the check (1.) and the validation (2.)
    SkipExistingCheck        : bool  = False # True: step 1 doesn't stop the script, even if the existing prototypes already reach RequiredAccuracy_Percent
    IgnoreValidation         : bool  = False # True: step 2 doesn't stop the script, the new prototypes are added even if they are below RequiredAccuracy_Percent

def AddNewMaterialPrototype(Controls: ControlVariables) -> str | None:
    # Returns the new network folder, or None if no prototypes were added
    BaseDirectory = os.path.dirname(os.path.abspath(__file__))
    PrototypePath = Controls.PrototypePath or f"{BaseDirectory}/DataSets/AdditionalPrototypes"
    OutputPath    = f"{BaseDirectory}/Outputs/{Controls.NetworkRun}+{Controls.Material}"
    shutil.rmtree(OutputPath, ignore_errors=True) # a rerun must not leave the network files of an earlier, successful run behind
    LogFile       = LOG.NoteFile(f"{OutputPath}/AddNewMaterialPrototype.log")
    LogFile.W(f"Adding prototypes of {Controls.Material} to the network {Controls.NetworkRun}\n{json.dumps(asdict(Controls), indent=4)}")

    Classifier = CI.ImageClassifier(Controls.NetworkRun)
    States     = set(Classifier.PrototypeStates)
    Folders    = sorted(Entry for Entry in os.listdir(PrototypePath) if Entry.split("_")[0] == Controls.Material and os.path.isdir(f"{PrototypePath}/{Entry}"))
    if not Folders:
        raise FileNotFoundError(f"No folders {Controls.Material}_<Reference>_<State> in {PrototypePath}")
    for Folder in Folders:
        if TEST.SurfaceState(Folder) not in States or Folder.count("_") > 2:
            raise ValueError(f"{Folder}: the folder name has to be {Controls.Material}_<Reference>_<State> or {Controls.Material}_<State> with a state of the network {sorted(States)}")

    # Tile embeddings of every folder. Group: the image of a tile, or the half of the image if the folder has only one image (for the validation).
    # The DINO features, names and thumbnails of the tiles are kept for the PCA plots
    Embeddings, Groups, Halved, Tiles = {}, {}, set(), {"Members": [], "Names": [], "Previews": [], "Features": []}
    LogFile.W("\nReference images")
    for Folder in Folders:
        Files = CI.FindImages(f"{PrototypePath}/{Folder}")
        if not Files:
            raise FileNotFoundError(f"No images in {PrototypePath}/{Folder}")
        if len(Files) == 1:
            Halved.add(Folder)
        FolderEmbeddings, FolderGroups = [], []
        for Index, File in enumerate(Files):
            with Image.open(File) as Opened:
                Picture = Opened.convert("RGB")
            TileEmbeddings, Boxes, Features = Classifier.EmbedTiles(Picture, File)
            FolderEmbeddings.append(TileEmbeddings)
            FolderGroups += [Index if len(Files) > 1 else int((Box[0] + Box[2]) / 2 >= Picture.width / 2) for Box in Boxes]
            Tiles["Members"]  += [Folder] * len(Boxes)
            Tiles["Names"]    += [f"{os.path.basename(File)}, tile at x = {Box[0]}, y = {Box[1]}" for Box in Boxes]
            Tiles["Previews"] += [IPCA.Thumbnail(Picture.crop(Box)) for Box in Boxes]
            Tiles["Features"].append(Features)
            LogFile.W(f"  {Folder:<45} {os.path.basename(File)}: {len(Boxes)} tiles")
            print(f"{Folder}: {os.path.basename(File)} ({len(Boxes)} tiles)")
        Embeddings[Folder], Groups[Folder] = torch.cat(FolderEmbeddings), torch.tensor(FolderGroups)
    Tiles["Embeddings"], Tiles["Features"] = torch.cat(list(Embeddings.values())), torch.cat(Tiles["Features"])

    def Accuracies(Folder, Prototypes, Classes, Tiles):
        # Share of tiles [%] assigned to a prototype with the right surface state, and to the prototype of the folder itself
        Predicted = [Classes[Label] for Label in NN.EuclideanDistance(Tiles, Prototypes).argmin(1).tolist()]
        return (100 * sum(TEST.SurfaceState(Class) == TEST.SurfaceState(Folder) for Class in Predicted) / len(Predicted),
                100 * sum(Class == Folder for Class in Predicted) / len(Predicted))

    # The network as it is written in step 3: the existing prototypes are kept, prototypes of classes with the same name (an earlier addition)
    # are replaced, and every folder adds the mean of all its tiles
    Existing   = [Classifier.LookUpTable[Label] for Label in range(len(Classifier.LookUpTable))]
    Classes    = [Class for Class in Existing if Class not in Folders] + Folders
    Kept       = [Classifier.Network.Prototypes[Index] for Index, Class in enumerate(Existing) if Class not in Folders]
    Prototypes = torch.stack(Kept + [Embeddings[Folder].mean(0) for Folder in Folders])
    PlotNewPrototypes(LogFile, Classifier, Classes, Prototypes, Tiles, OutputPath, f"{Controls.NetworkRun}+{Controls.Material}")

    # 1. Check with the existing prototypes
    CheckState = {Folder: Accuracies(Folder, Classifier.Network.Prototypes, Existing, Embeddings[Folder])[0] for Folder in Folders}
    LogFile.W("\n1. Tiles with the correct surface state with the existing prototypes")
    for Folder in Folders:
        LogFile.W(f"  {Folder:<45} {CheckState[Folder]:6.2f} %")
    if min(CheckState.values()) >= Controls.RequiredAccuracy_Percent and not Controls.SkipExistingCheck:
        return Finish(LogFile, None, f"The existing prototypes classify at least {Controls.RequiredAccuracy_Percent} % of the tiles of every reference correctly. "
                                     f"No prototypes added (set SkipExistingCheck / --skip-existing-check to add them anyway)")

    PrototypesWithout =lambda Excluded, Group: torch.stack(Kept + [Embeddings[Folder][(Groups[Folder] != Group) | (Folder != Excluded)].mean(0) for Folder in Folders])

    # 2. Validation: every group (image or image half) is held out once
    LogFile.W(f"\n2. Validation: held out tiles assigned to a prototype with the correct surface state / to the prototype of their own folder")
    Passed = True
    for Folder in Folders:
        Results = [(Accuracies(Folder, PrototypesWithout(Folder, Group), Classes, Embeddings[Folder][Groups[Folder] == Group]), int((Groups[Folder] == Group).sum()))
                   for Group in Groups[Folder].unique().tolist()]
        StateAccuracy     = sum(Accuracy[0] * Tiles for Accuracy, Tiles in Results) / len(Groups[Folder])
        ReferenceAccuracy = sum(Accuracy[1] * Tiles for Accuracy, Tiles in Results) / len(Groups[Folder])
        Passed           &= StateAccuracy >= Controls.RequiredAccuracy_Percent
        LogFile.W(f"  {Folder:<45} surface state {StateAccuracy:6.2f} %, own prototype {ReferenceAccuracy:6.2f} % "
                  f"({len(Results)} held out {'image halves' if Folder in Halved else 'images'})")
    if not Passed:
        Message = f"Validation failed: less than {Controls.RequiredAccuracy_Percent} % of the held out tiles of a reference have the correct surface state. "
        if not Controls.IgnoreValidation:
            return Finish(LogFile, None, Message + "No prototypes added - check the reference images or add more (or set IgnoreValidation / --ignore-validation)")
        print(f"WARNING! - {Message}The prototypes are added anyway (IgnoreValidation)")
        LogFile.W(f"\nWARNING! - {Message}The prototypes are added anyway (IgnoreValidation)")

    # 3. New network folder with all tiles of every folder
    shutil.copy(f"{Classifier.RunPath}/TrainedProtoNet.pth", f"{OutputPath}/TrainedProtoNet.pth")
    torch.save(Prototypes, f"{OutputPath}/PrototypeEmbedding.pth")
    with open(f"{OutputPath}/Labelmap.json", "w") as LabelMapFile:
        json.dump({Class: Label for Label, Class in enumerate(Classes)}, LabelMapFile)
    with open(f"{Classifier.RunPath}/Settings.json") as SettingsFile:
        Settings = json.load(SettingsFile)
    Settings["AddedPrototypes"] = Settings.get("AddedPrototypes", []) + [{"Material": Controls.Material, "BaseNetwork": Classifier.RunPath, "Folders": Folders, "ValidationPassed": Passed}]
    with open(f"{OutputPath}/Settings.json", "w") as SettingsFile:
        json.dump(Settings, SettingsFile, indent=4)
    return Finish(LogFile, OutputPath, f"Added {len(Folders)} prototypes, the network now has {len(Classes)}: {OutputPath}")

def PlotNewPrototypes(LogFile, Classifier, Classes: list[str], Prototypes: torch.Tensor, Tiles: dict, OutputPath: str, RunName: str):
    '''
    Shows where the reference tiles and the new prototypes lie compared with the training materials, in every run (also if no prototypes are added):
      PCA_NewPrototypes.png   the embeddings
      PCA_NewPrototypes.html  interactive, embeddings and DINO features, with a preview of every reference tile
    The test set of the training data is classified with the network as it is written in step 3 (existing and new prototypes),
    see TestSet.log and ConfusionMatrix.csv. The PCAs are fitted on this test set like the PCAs of the training run, so the positions are comparable.
    '''
    with open(f"{Classifier.RunPath}/Settings.json") as SettingsFile:
        Settings = json.load(SettingsFile)
    BaseDirectory = os.path.dirname(os.path.abspath(__file__))
    Controls = SimpleNamespace(DataSet=Settings["DataSet"], DataSetPath=f"{BaseDirectory}/DataSets/{Settings['DataSet']}", FeatureExtractor=Settings["FeatureExtractor"],
                               WriteDetailedDebugInfo=False, OutputPath=OutputPath, Seed=Settings["Seed"], RunName=RunName)
    TestLog = LOG.NoteFile(f"{OutputPath}/TestSet.log")
    TestLog.W(f"Test set of {Settings['DataSet']}, classified with the existing and the new prototypes ({RunName})")
    DataSet = DATA.DataSet()
    DataSet.LoadDataSet(TestLog, Controls)

    # Test set, classified with the existing and the new prototypes
    Members = [f"{Sample}_{Class}" for Sample in DataSet.Samples for Class in DataSet.Classes]
    LookUp  = dict(enumerate(Classes))
    Network = SimpleNamespace(Encoder=Classifier.Network.Encoder, Prototypes=Prototypes)
    Handler = TEST.TestHandler(DataSet, SimpleNamespace(LabelMap={Class: Label for Label, Class in LookUp.items()}, LookUpTable=LookUp,
                               GroupOf={Member: Member if Settings.get("ClassGrouping", "MaterialAndState") == "MaterialAndState" else TEST.SurfaceState(Member) for Member in Members}), Network)
    print("\nTest set of the training materials with the existing and the new prototypes:")
    TEST.NetworkTestApplication(TestLog, Controls, Network, Handler)
    TestLog.Close()
    Data, Folders = Handler.PreparedData, list(dict.fromkeys(Tiles["Members"]))
    ToNew   = sum(LookUp[Label] in Folders for Label in Data["Classifications"])
    LogFile.W(f"\nTest set of the training materials with the existing and the new prototypes: "
              f"{ToNew} of {len(Data['Classifications'])} test images are closest to a new prototype (details in TestSet.log and ConfusionMatrix.csv)")

    # Reference tiles, classified with the existing and the new prototypes
    Labels, _, StateProbabilities = TEST.ClassifyEmbeddings(Tiles["Embeddings"], Prototypes, [TEST.SurfaceState(Class) for Class in Classes])

    EmbeddingPca = PCA(n_components=2).fit(Data["Embeddings"].numpy())
    FeaturePca   = PCA(n_components=2).fit(Data["FeatureTensor"].numpy())
    Views = [{"Name": "Embeddings", "Coordinates": EmbeddingPca.transform(Data["Embeddings"].numpy()), "Variance": EmbeddingPca.explained_variance_ratio_,
              "Prototypes": EmbeddingPca.transform(Prototypes.numpy())},
             {"Name": "DINO features", "Coordinates": FeaturePca.transform(Data["FeatureTensor"].numpy()), "Variance": FeaturePca.explained_variance_ratio_, "Prototypes": None}]
    TilePoints = EmbeddingPca.transform(Tiles["Embeddings"].numpy())
    IPCA.WriteInteractivePCA(Controls, Handler, Views, Settings.get("InteractivePCAImages", 1000), FileName="PCA_NewPrototypes.html",
                             Title=f"Reference tiles and new prototypes - {RunName}",
                             Extra={"Members": Tiles["Members"], "Names": Tiles["Names"], "Previews": Tiles["Previews"], "Classifications": Labels,
                                    "StateProbabilities": StateProbabilities, "Coordinates": [TilePoints, FeaturePca.transform(Tiles["Features"].numpy())]})

    # Static plot of the embeddings, with the same colours and markers as the interactive one
    Style = LOG.ClassStyle(DataSet.Samples, DataSet.Classes, Members + Classes + Folders)
    Marks = lambda Name: dict(zip(("c", "marker"), Style(Name)))
    Figure, Axis = plt.subplots(figsize=(12, 8))
    TestMembers = np.array(Data["Members"])
    for Member in Members:
        Points = Views[0]["Coordinates"][TestMembers == Member]
        Axis.scatter(Points[:, 0], Points[:, 1], s=12, alpha=0.25, linewidths=0, zorder=1, label=f"Test set: {Member}", **Marks(Member))
    TileMembers = np.array(Tiles["Members"])
    for Folder in Folders:
        Points = TilePoints[TileMembers == Folder]
        Axis.scatter(Points[:, 0], Points[:, 1], s=30, edgecolors="white", linewidths=0.5, alpha=0.85, zorder=2, label=f"Reference tiles: {Folder} ({len(Points)})", **Marks(Folder))
    # Prototypes: large, with a red outline. Prototypes added in an earlier run have no points of their own, so they get a legend entry
    for Name, Point in zip(Classes, Views[0]["Prototypes"]):
        Earlier = LOG.MaterialOf(Name) not in DataSet.Samples + [""] and Name not in Folders
        Axis.scatter(*Point, s=180, edgecolors="#d00000", linewidths=1.6, zorder=4, label=f"Prototype added earlier: {Name}" if Earlier else None, **Marks(Name))
    Axis.scatter([], [], s=180, c="#8a8a8a", marker="o", edgecolors="#d00000", linewidths=1.6, label="Prototype (existing and new)")

    Variance = Views[0]["Variance"]
    Axis.set_title("New prototypes in the embedding space (PCA of the test set embeddings)", fontsize=16, color="#1f1f1f")
    Axis.set_xlabel(f"PC1 ({100*Variance[0]:.2f}% of original variance)", fontsize=13)
    Axis.set_ylabel(f"PC2 ({100*Variance[1]:.2f}% of original variance)", fontsize=13)
    Axis.grid(True, color="#e3e3e3", linewidth=0.8)
    Axis.set_axisbelow(True)
    Legend = Axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=10)
    for Handle in Legend.legend_handles:
        Handle.set_alpha(1)
    Figure.savefig(f"{OutputPath}/PCA_NewPrototypes.png", dpi=200, bbox_inches="tight")
    plt.close(Figure)

def Finish(LogFile, OutputPath: str | None, Message: str) -> str | None:
    print(Message)
    LogFile.W(f"\n{Message}")
    LogFile.Close()
    return OutputPath

def Main():
    Defaults = ControlVariables()
    Parser   = argparse.ArgumentParser(description="Add prototypes of a new material to a trained network.")
    Parser.add_argument("Material", nargs="?", default=Defaults.Material, help="Material name, the start of its folder names in src/DataSets/AdditionalPrototypes/")
    Parser.add_argument("--network", default=Defaults.NetworkRun, help="Training run in src/Outputs/")
    Parser.add_argument("--skip-existing-check", action="store_true", help=f"Step 1: continue even if the existing prototypes already classify at least {Defaults.RequiredAccuracy_Percent:g} %% of the reference tiles correctly")
    Parser.add_argument("--ignore-validation", action="store_true", help=f"Step 2: add the new prototypes even if they are below {Defaults.RequiredAccuracy_Percent:g} %% in the validation")
    Arguments = Parser.parse_args()
    if not Arguments.Material or not Arguments.network:
        Parser.error("Give the material and the network (command line or ControlVariables)")
    AddNewMaterialPrototype(ControlVariables(Material=Arguments.Material, NetworkRun=Arguments.network,
                                             SkipExistingCheck=Arguments.skip_existing_check or Defaults.SkipExistingCheck,
                                             IgnoreValidation=Arguments.ignore_validation or Defaults.IgnoreValidation))

if __name__ == "__main__":
    Main()
