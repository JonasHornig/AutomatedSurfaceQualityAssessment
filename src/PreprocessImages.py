import time
StartTime = time.time()

import h5py
import os

from dataclasses import dataclass, field
from PIL import Image

from tqdm import tqdm
from torchvision.transforms import InterpolationMode

import Modules.DataSetHandling as DATA
import Modules.NeuralNetwork as NN
import Modules.Logging as LOG

@dataclass
class ControlVariables:
    # Inputs
    DataSet : str       = "CompleteDataSet" #  --  DummyDataSet  --  SmallDataSet  --  CompleteDataSet
    Samples : list[str] = field(default_factory=lambda: [])

    # Flags
    #  --  True  --  False  --
    WriteDetailedDebugInfo : bool = True
    ProcessAllImages       : bool = True

    # Image handling
    TargetImageSize : tuple = (224,224)
    TargetBatchSize : int   = 64

    # Parameter initialisation
    NumberOfImages : int = 0
    DataSetPath    : str = ""

    def __post_init__(self):
        self.DataSetPath = f"DataSets/{self.DataSet}"

def Main(LogFile):
    LogFile.W("\n*****************************\n*  Data set pre-processing  *\n*       using DINOv2        *\n*****************************")

    Controls  = ControlVariables()
    DinoModel = NN.DINOv2Model()
    MetaData  = {}
    LogFile.W("\nLoading DINOv2 model complete.")

    if Controls.ProcessAllImages:
        Samples = [Entry for Entry in os.listdir(Controls.DataSetPath) if os.path.isdir(os.path.join(Controls.DataSetPath, Entry))]
    else:
        Samples = Controls.Samples
    print("Load images from data set.")
    DataSet = DATA.DataSet(Controls.TargetImageSize)
    DataSet.LoadDataSet(LogFile, Controls, ForceLoadImages=True)

    LogFile.W("\nProcessing images")
    with tqdm(total=DataSet.NumberOfImages, desc=f"Extracting features", unit=" img") as ProgressBar:
        for Sample in DataSet.Images:
            for Class in DataSet.Images[Sample]:
                for Mode in DataSet.Images[Sample][Class]:
                    LogFile.W(f"{Sample}, {Class}, {Mode}, ", NewLine=False)
                    #Images = [DataSet.ValidateImageSize(Image.open(ImagePath)) for ImagePath in DataSet.Images[Sample][Class][Mode]["Names"]]
                    DataSet.Images[Sample][Class][Mode]["Images"] = []
                    for ImagePath in DataSet.Images[Sample][Class][Mode]["Names"]:
                        PilImage       = Image.open(f"{Controls.DataSetPath}/Images/{Sample}/{Class}/{Mode}/{ImagePath}")
                        ValidatedImage = DataSet.ValidateImageSize(PilImage)
                        DataSet.Images[Sample][Class][Mode]["Images"].append(ValidatedImage)
                    DinoModel.ApplyNetwork(DataSet.Images[Sample][Class][Mode], LogFile, ProgressBar, Controls.WriteDetailedDebugInfo)
                    for PilImage in DataSet.Images[Sample][Class][Mode]["Images"]:
                        PilImage.close()

    for Sample in DataSet.Images:
        with h5py.File(f"{Controls.DataSetPath}/{Controls.DataSet}_{Sample}_DINOv2Features.h5", "w") as HdfFile:
            for Class in DataSet.Images[Sample]:
                for Mode in DataSet.Images[Sample][Class]:
                    HdfFile.create_dataset(f"{Class}/{Mode}/Names"       , data = DataSet.Images[Sample][Class][Mode]["Names"])
                    HdfFile.create_dataset(f"{Class}/{Mode}/DinoFeatures", data = DataSet.Images[Sample][Class][Mode]["DinoFeatures"], compression = "gzip", compression_opts = 4)
            HdfMetaData = HdfFile.require_group("MetaData")
            for Key in MetaData.keys():
                HdfMetaData.attrs[Key] = MetaData[Key]

            LOG.LogHdf5Write(LogFile, Controls, HdfFile)

if __name__ == "__main__":
    print("")
    LogFile = LOG.NoteFile(f"Outputs/Preprocessing.log")
    Main(LogFile)

    LogFile.W(f"\nPre-processing complete")
    Duration = time.time() - StartTime
    Hours = int(Duration // 3600)
    Minutes = int((Duration % 3600) // 60)
    Seconds = int(Duration % 60)
    print(f"\nRuntime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}")
    LogFile.W(f"Runtime: {Hours:02d}:{Minutes:02d}:{Seconds:02d}", NewLine=False)
    LogFile.Close()