import time
StartTime = time.time()

import h5py
import os

from dataclasses import dataclass
from PIL import Image

from tqdm import tqdm

import Modules.DataSetHandling as DATA
import Modules.NeuralNetwork as NN
import Modules.Logging as LOG

@dataclass
class ControlVariables:
    # Inputs
    DataSet : str       = "CompleteDataSet" #  --  DummyDataSet  --  SmallDataSet  --  CompleteDataSet

    # Feature extractor, see NeuralNetwork.DINOV3_MODELS. The weights are located in <Repository>/src/Models/
    FeatureExtractor : str = "DINOv3" #  --  "DINOv3" (ViT-B/16)  --  "DINOv3-H16plus" (ViT-H+/16, about 10 times slower)  --  "DINOv1" (legacy, provided as files only)

    # Flags
    #  --  True  --  False  --
    WriteDetailedDebugInfo : bool = False # True: also logs the feature vector of every image (large log files)
    ResumeExtraction       : bool = True  # True: classes that are already complete in the feature file are kept and skipped, so an interrupted run continues where it stopped. False: the feature files are written anew

    # Image handling
    TargetImageSize : tuple = (224,224)
    MaxBatchSize    : int   = 16 # Maximum number of images passed through DINOv3 at once. Limits the memory use, which matters for the large ViT-H+ model

    def __post_init__(self):
        # Paths are resolved from the location of this script, so it can be run from any working directory
        BaseDirectory        = os.path.dirname(os.path.abspath(__file__))
        self.DataSetPath     = f"{BaseDirectory}/DataSets/{self.DataSet}"
        self.OutputPath      = f"{BaseDirectory}/Outputs"
        self.ModelDirectory  = f"{BaseDirectory}/Models"

def IsComplete(HdfFile, Group: str, Names: list[str]) -> bool:
    # A class is complete if its features and names were written and the names match the images of the class
    if f"{Group}/Names" not in HdfFile or f"{Group}/DinoFeatures" not in HdfFile:
        return False
    StoredNames = [Name.decode("utf-8") for Name in HdfFile[f"{Group}/Names"][:]] # type: ignore
    return StoredNames == Names and HdfFile[f"{Group}/DinoFeatures"].shape[0] == len(Names) # type: ignore

def Main(LogFile, Controls):
    LogFile.W(f"\n*****************************\n*  Data set pre-processing  *\n*****************************\nData set: {Controls.DataSet}, feature extractor: {Controls.FeatureExtractor}")

    DinoModel = NN.DINOv3Model(Controls.FeatureExtractor, Controls.ModelDirectory)
    MetaData  = {"FeatureExtractor": DinoModel.HubEntryPoint, "Weights": DinoModel.WeightsName, "HubRepository": NN.DINOV3_HUB_REPOSITORY}
    LogFile.W(f"\nLoading {DinoModel.HubEntryPoint} complete.")

    print("Load images from data set.")
    DataSet = DATA.DataSet(Controls.TargetImageSize)
    DataSet.LoadDataSet(LogFile, Controls, ForceLoadImages=True)

    LogFile.W("\nProcessing images")
    with tqdm(total=DataSet.TotalNumberOfImages, desc=f"Extracting features", unit=" img") as ProgressBar:
        for Sample in DataSet.Images:
            # Each class is written to the feature file of its sample as soon as it is finished, so an interrupted run only loses the class in progress
            with h5py.File(f"{Controls.DataSetPath}/{Controls.DataSet}_{Sample}_{Controls.FeatureExtractor}Features.h5", "a" if Controls.ResumeExtraction else "w") as HdfFile:
                HdfMetaData = HdfFile.require_group("MetaData")
                for Key in MetaData.keys():
                    HdfMetaData.attrs[Key] = MetaData[Key]

                for Class in DataSet.Images[Sample]:
                    for Mode in DataSet.Images[Sample][Class]:
                        Images = DataSet.Images[Sample][Class][Mode]
                        Group  = f"{Class}/{Mode}"
                        if IsComplete(HdfFile, Group, Images["Names"]):
                            LogFile.W(f"{Sample}, {Class}, {Mode}: already extracted, skipped")
                            ProgressBar.update(len(Images["Names"]))
                            continue
                        if Group in HdfFile:
                            del HdfFile[Group] # incomplete class of an interrupted run

                        LogFile.W(f"{Sample}, {Class}, {Mode}, ", NewLine=False)
                        Images["Images"] = []
                        for ImagePath in Images["Names"]:
                            with Image.open(f"{Controls.DataSetPath}/Images/{Sample}/{Class}/{Mode}/{ImagePath}") as PilImage:
                                ValidatedImage = DataSet.ValidateImage(PilImage, ImagePath, LogFile)
                            Images["Images"].append(ValidatedImage)
                        DinoModel.ApplyNetwork(Images, LogFile, ProgressBar, Controls.WriteDetailedDebugInfo, Controls.MaxBatchSize)
                        for PilImage in Images["Images"]:
                            PilImage.close()

                        # The names are written last, so they mark the class as complete
                        HdfFile.create_dataset(f"{Group}/DinoFeatures", data = Images["DinoFeatures"], compression = "gzip", compression_opts = 4)
                        HdfFile.create_dataset(f"{Group}/Names"       , data = Images["Names"])
                        HdfFile.flush()
                        Images["Images"], Images["DinoFeatures"] = [], None # free the memory before the next class

                LOG.LogHdf5Write(LogFile, Controls, HdfFile)

    if len(DataSet.ChangedImages) > 0:
        LogFile.W(f"\nWARNING! - {len(DataSet.ChangedImages)} images did not match the target size of [{Controls.TargetImageSize[0]} x {Controls.TargetImageSize[1]}].")

if __name__ == "__main__":
    print("")
    Controls = ControlVariables()
    LogFile  = LOG.NoteFile(f"{Controls.OutputPath}/Preprocessing_{Controls.DataSet}_{Controls.FeatureExtractor}.log")
    Main(LogFile, Controls)

    LogFile.W(f"\nPre-processing complete")
    Duration = time.time() - StartTime
    Hours = int(Duration // 3600)
    Minutes = int((Duration % 3600) // 60)
    Seconds = int(Duration % 60)
    print(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}")
    LogFile.W(f"Runtime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}", NewLine=False)
    LogFile.Close()