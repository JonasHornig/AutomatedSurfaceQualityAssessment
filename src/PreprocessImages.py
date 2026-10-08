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

    # Image handling
    TargetImageSize : tuple = (224,224)
    MaxBatchSize    : int   = 64 # Maximum number of images passed through DINOv3 at once, limits the memory use

    def __post_init__(self):
        # Paths are resolved from the location of this script, so it can be run from any working directory
        BaseDirectory        = os.path.dirname(os.path.abspath(__file__))
        self.DataSetPath     = f"{BaseDirectory}/DataSets/{self.DataSet}"
        self.OutputPath      = f"{BaseDirectory}/Outputs"
        self.ModelDirectory  = f"{BaseDirectory}/Models"

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
            for Class in DataSet.Images[Sample]:
                for Mode in DataSet.Images[Sample][Class]:
                    LogFile.W(f"{Sample}, {Class}, {Mode}, ", NewLine=False)
                    DataSet.Images[Sample][Class][Mode]["Images"] = []
                    for ImagePath in DataSet.Images[Sample][Class][Mode]["Names"]:
                        with Image.open(f"{Controls.DataSetPath}/Images/{Sample}/{Class}/{Mode}/{ImagePath}") as PilImage:
                            ValidatedImage = DataSet.ValidateImage(PilImage, ImagePath, LogFile)
                        DataSet.Images[Sample][Class][Mode]["Images"].append(ValidatedImage)
                    DinoModel.ApplyNetwork(DataSet.Images[Sample][Class][Mode], LogFile, ProgressBar, Controls.WriteDetailedDebugInfo, Controls.MaxBatchSize)
                    for PilImage in DataSet.Images[Sample][Class][Mode]["Images"]:
                        PilImage.close()

            # Each sample is written as soon as it is finished, so an interrupted run only loses the sample in progress
            with h5py.File(f"{Controls.DataSetPath}/{Controls.DataSet}_{Sample}_{Controls.FeatureExtractor}Features.h5", "w") as HdfFile:
                for Class in DataSet.Images[Sample]:
                    for Mode in DataSet.Images[Sample][Class]:
                        HdfFile.create_dataset(f"{Class}/{Mode}/Names"       , data = DataSet.Images[Sample][Class][Mode]["Names"])
                        HdfFile.create_dataset(f"{Class}/{Mode}/DinoFeatures", data = DataSet.Images[Sample][Class][Mode]["DinoFeatures"], compression = "gzip", compression_opts = 4)
                HdfMetaData = HdfFile.require_group("MetaData")
                for Key in MetaData.keys():
                    HdfMetaData.attrs[Key] = MetaData[Key]

                LOG.LogHdf5Write(LogFile, Controls, HdfFile)

    if DataSet.NumberOfResizedImages > 0:
        LogFile.W(f"\nWARNING! - {DataSet.NumberOfResizedImages} images did not match the target size {Controls.TargetImageSize} and were resized. Check the tiling.")
        print(f"WARNING! - {DataSet.NumberOfResizedImages} images were resized. See log file.")

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