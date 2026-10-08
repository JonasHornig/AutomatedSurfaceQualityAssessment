import os
import h5py
import torch

from PIL import Image

from . import AuxiliaryCode as AUX
from . import Logging as LOG

class DataSet():
    '''
    Master image handler class
    Assumes: 
        - Sorted data set. An image can be found under its Directory MyDataSet/Images/ImageSample/ImageClass/UserMode/ImageName.tif
    
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

        self.ProcessedDataSet  : bool = False
        self.Overwriteh5pyFile : bool = False

        self.DataSetPath : str  = ""
        self.DataSetName : str  = ""
        self.TotalNumberOfImages    : int = 0

    def ValidateImageSize(self, Image: Image.Image) -> Image.Image:
        if Image.size != self.TargetImageSize:
            Image = Image.resize(self.TargetImageSize)
        return Image

    def LoadDataSet(self, LogFile, Controls, ForceLoadImages: bool = False):
        # Assumes that the data set is balanced, meaning the same classes and modes for each input sample.

        LogFile.W("\nLoading data set\n==================")
        LogFile.W("*---------------------------------------------------*")
        LogFile.W("| Class      | Sample     | Mode       | Number     |")
        LogFile.W("|            |            |            | of images  |")
        LogFile.W("|------------|------------|------------|------------|")

        self.DataSetPath = Controls.DataSetPath
        self.DataSetName = Controls.DataSet

        self.HdfFiles = [Entry for Entry in os.listdir(self.DataSetPath) if Entry.endswith(".h5") and Entry.startswith(self.DataSetName + "_")]

        if self.HdfFiles and not ForceLoadImages:
            self.LoadDataSetFromHdfFile(LogFile, WriteDetailedDebugInfo=Controls.WriteDetailedDebugInfo)
        else:
            self.LoadDataSetImages(LogFile, WriteDetailedDebugInfo=Controls.WriteDetailedDebugInfo)

    def LoadDataSetFromHdfFile(self, LogFile, WriteDetailedDebugInfo: bool = True):
        Warnings = []
        
        for HdfFileName in self.HdfFiles:
            with h5py.File(f"{self.DataSetPath}/{HdfFileName}", "r") as HdfFile:
                Sample = HdfFileName.split("_")[1]
                self.Samples.append(Sample)
                AUX.AppendDictionary(self.Images, Sample, "dict")

                if self.Classes and not all(Class in list(HdfFile.keys()) for Class in self.Classes):
                    Warnings.append(f"Classes for sample {self.Samples[-2]}: {self.Classes} - Classes for sample {self.Samples[-1]}: {list(HdfFile.keys())}")
                else:
                    self.Classes = list(HdfFile.keys()) # type: ignore
                if "MetaData" in self.Classes:
                    self.Classes.remove("MetaData")
                
                for Class in self.Classes:
                        AUX.AppendDictionary(self.Images[Sample], Class, "dict")
                        for Mode in list(HdfFile[Class].keys()): # type: ignore
                            AUX.AppendDictionary(self.Images[Sample][Class], Mode, "dict")
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
        LogFile.W(f"Loaded from HDF5 file")

        if WriteDetailedDebugInfo:
            LogFile.W("\nStructure of the read image dictionary:")
            LOG.LogDict(LogFile, self.Images)

        self.ProcessedDataSet = True

    def LoadDataSetImages(self, LogFile, Samples: list = [], WriteDetailedDebugInfo: bool = True):
        BasePath = f"{self.DataSetPath}/Images"
        self.NumberOfImages = 0

        if not Samples:
            self.Samples = [Entry for Entry in os.listdir(BasePath) if os.path.isdir(os.path.join(BasePath, Entry))]
        else:
            self.Samples = Samples
        for Sample in self.Samples:
            AUX.AppendDictionary(self.Images, Sample, "dict")
            SamplePath = f"{BasePath}/{Sample}"

            self.Classes = [Entry for Entry in os.listdir(SamplePath) if os.path.isdir(os.path.join(SamplePath, Entry))]
            for Class in self.Classes:
                AUX.AppendDictionary(self.Images[Sample], Class, "dict")
                ClassPath = f"{SamplePath}/{Class}"

                self.Modes = [Entry for Entry in os.listdir(ClassPath) if os.path.isdir(os.path.join(ClassPath, Entry))]
                for Mode in self.Modes:
                    AUX.AppendDictionary(self.Images[Sample][Class], Mode, "dict")
                    ModePath = f"{ClassPath}/{Mode}"

                    ImageNames = [Entry for Entry in os.listdir(ModePath) if Entry.endswith(".tif")]
                    AUX.AppendDictionary(self.Images[Sample][Class][Mode], "Images", "list")
                    AUX.AppendDictionary(self.Images[Sample][Class][Mode], "Names" , "list")
                    for ImageName in ImageNames:
                        ImagePath      = f"{ModePath}/{ImageName}"
                        '''
                        PilImage       = Image.open(ImagePath)
                        ValidatedImage = self.ValidateImageSize(PilImage)
                        self.Images[Sample][Class][Mode]["Images"].append(ValidatedImage)
                        '''
                        self.Images[Sample][Class][Mode]["Names"].append(ImageName)
                    self.NumberOfImages += len(ImageNames)
                
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

    def GetParameters(self) -> dict:
        return {
            "Samples"                : self.Samples                ,
            "Classes"                : self.Classes                ,
            "Images"                 : self.Images                 }