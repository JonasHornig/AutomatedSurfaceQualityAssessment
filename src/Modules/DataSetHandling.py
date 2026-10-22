import os
import h5py
import torch

from PIL import Image

from . import Logging as LOG

class DataSet():
    '''
    Master image handler class
    Assumes: 
        - Sorted data set. An image can be found under its Directory <DATASETNAME>/Images/ImageSample/ImageClass/UsageMode/ImageName.tif
        - Balanced data set. Meaning the same classes and modes for each input sample.
    
    Dictionary order:
        {<Sample> : 
            {<Class> : 
                {<Mode> : 
                    {<Images> = [ ... PIL Images ... ] ; <Names> = [ ... Image Names ... ]
                    }
                }
            }
        }
    '''

    def __init__(self, ImageSize: tuple = (224,224)):
        self.TargetImageSize : tuple     = ImageSize

        self.Images  : dict      = {}
        self.Classes : list[str] = []
        self.Samples : list[str] = []
        self.Modes   : list[str] = []

        # Variable initialization
        self.DataSetPath : str  = ""
        self.DataSetName : str  = ""
        self.TotalNumberOfImages   : int = 0
        self.ChangedImages         : list[str] = []

        # Flags
        self.ImagePropertiesChanged : bool = False

    @property
    def FeatureDimension(self) -> int:
        Sample, Class = self.Samples[0], self.Classes[0]
        Mode          = next(iter(self.Images[Sample][Class]))
        return self.Images[Sample][Class][Mode]["DinoFeatures"].shape[1]

    def ValidateImage(self, Image: Image.Image, ImageName: str, LogFile) -> Image.Image:
        # DINOv3 expects 3 channel RGB input. convert() always returns a copy, so the opened file can be closed afterwards.
        ValidatedImage = Image.convert("RGB")

        # Tiles are cut from the microscope images at full resolution and should already match the target size.
        # A mismatch indicates a tiling error, as resizing would blur the scratches the network is supposed to detect.
        if ValidatedImage.size != self.TargetImageSize:
            self.ChangedImages.append(ImageName)
            if not self.ImagePropertiesChanged:
                self.ImagePropertiesChanged = True
            ValidatedImage = ValidatedImage.resize(self.TargetImageSize)
        return ValidatedImage

    def LoadDataSet(self, LogFile, Controls, ForceLoadImages: bool = False):
        LOG.DataSetHeader(LogFile)

        self.DataSetPath = Controls.DataSetPath
        self.DataSetName = Controls.DataSet

        # All directory listings are sorted. os.listdir has no defined order (alphabetical on Windows, arbitrary on Linux),
        # and the order of samples and classes determines the label map and the classes drawn for each episode.
        self.HdfFiles = [Entry for Entry in sorted(os.listdir(self.DataSetPath)) if Entry.endswith(f"_{Controls.FeatureExtractor}Features.h5") and Entry.startswith(self.DataSetName + "_")]

        if ForceLoadImages:
            self.LoadDataSetImages(LogFile, WriteDetailedDebugInfo=Controls.WriteDetailedDebugInfo)
        elif self.HdfFiles:
            self.LoadDataSetFromHdfFile(LogFile, WriteDetailedDebugInfo=Controls.WriteDetailedDebugInfo)
        else:
            # Feature extraction is a separate, slow step that needs the raw images and the DINO weights, so it is not started automatically
            Message = (f"No {Controls.FeatureExtractor} feature files found for {self.DataSetName} "
                       f"(expected {self.DataSetPath}/{self.DataSetName}_<Sample>_{Controls.FeatureExtractor}Features.h5).\n"
                       f"Run PreprocessImages.py with DataSet = \"{self.DataSetName}\" and FeatureExtractor = \"{Controls.FeatureExtractor}\" first.")
            LogFile.W(f"ERROR! - {Message}")
            raise FileNotFoundError(Message)

    def LoadDataSetImages(self, LogFile, WriteDetailedDebugInfo: bool = True):
        '''
        Collects only image names. The images are loaded dynamically when creating the batches for pre-processing to conserve memory usage.
        '''

        BasePath = f"{self.DataSetPath}/Images"

        self.Samples = [Entry for Entry in sorted(os.listdir(BasePath)) if os.path.isdir(os.path.join(BasePath, Entry))]
        for Sample in self.Samples:
            self.Images[Sample] = {}
            SamplePath = f"{BasePath}/{Sample}"
            self.Classes = [Entry for Entry in sorted(os.listdir(SamplePath)) if os.path.isdir(os.path.join(SamplePath, Entry))]

            for Class in self.Classes:
                self.Images[Sample][Class] = {}
                ClassPath = f"{SamplePath}/{Class}"
                self.Modes = [Entry for Entry in sorted(os.listdir(ClassPath)) if os.path.isdir(os.path.join(ClassPath, Entry))]

                for Mode in self.Modes:
                    ImageNames = [Entry for Entry in sorted(os.listdir(f"{ClassPath}/{Mode}")) if Entry.endswith(".tif")]
                    self.Images[Sample][Class][Mode] = {"Images": [], "Names": ImageNames}
                    self.TotalNumberOfImages += len(ImageNames)
                
                    LogFile.W(f"| {Class:<10} | {Sample:<10} | {Mode:<10} | {str(len(ImageNames)):<10} |")
        LogFile.W("*---------------------------------------------------*")
        LogFile.W(f"Number of samples: {len(self.Samples)} - {self.Samples}")
        LogFile.W(f"Number of classes: {len(self.Classes)} - {self.Classes}")
        LogFile.W(f"Number of modes  : {len(self.Modes)} - {self.Modes}")
        LogFile.W(f"Total number of images: {self.TotalNumberOfImages}")
        LogFile.W(f"Loaded from raw images")

        if WriteDetailedDebugInfo:
            LogFile.W("\nStructure of the read image dictionary:")
            LOG.LogDict(LogFile, self.Images)

    def LoadDataSetFromHdfFile(self, LogFile, WriteDetailedDebugInfo: bool = True):
        for HdfFileName in self.HdfFiles:
            with h5py.File(f"{self.DataSetPath}/{HdfFileName}", "r") as HdfFile:
                Sample = HdfFileName.split("_")[1]
                self.Samples.append(Sample)
                self.Images[Sample] = {}

                self.Classes = list(HdfFile.keys()) # type: ignore
                if "MetaData" in self.Classes:
                    self.Classes.remove("MetaData")
                
                for Class in self.Classes:
                    self.Images[Sample][Class] = {}
                    for Mode in list(HdfFile[Class].keys()): # type: ignore
                        self.Images[Sample][Class][Mode] = {
                            "DinoFeatures" : torch.tensor(HdfFile[f"{Class}/{Mode}/DinoFeatures"][:])               , # type: ignore
                            "Names"        : [Name.decode("utf-8") for Name in HdfFile[f"{Class}/{Mode}/Names"][:]] } # type: ignore
                        self.TotalNumberOfImages += len(self.Images[Sample][Class][Mode]["Names"])

                        LogFile.W(f"| {Class:<10} | {Sample:<10} | {Mode:<10} | {str(len(self.Images[Sample][Class][Mode]["Names"])):<10} |")

        LogFile.W("*---------------------------------------------------*")
        self.Samples = list(dict.fromkeys(self.Samples))
        LogFile.W(f"Number of samples: {len(self.Samples)} - {self.Samples}")
        self.Classes = list(dict.fromkeys(self.Classes))
        LogFile.W(f"Number of classes: {len(self.Classes)} - {self.Classes}")
        LogFile.W(f"Total number of images: {self.TotalNumberOfImages}")
        LogFile.W(f"Loaded from HDF5 files.")

        if WriteDetailedDebugInfo:
            LogFile.W("\nStructure of the read image dictionary:")
            LOG.LogDict(LogFile, self.Images)