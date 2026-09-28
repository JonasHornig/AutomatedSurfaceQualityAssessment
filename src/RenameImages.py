import os
import random
import string
import shutil

import Modules.AuxiallryCode as AUX

OldDataSetPath = "DataSets/SmallDataSet_Old"
NewDataSetPath = "DataSets/SmallDataSet"

print(os.getcwd())

Modes = ["Training", "Testing"]

for Mode in Modes:
    ModePath = f"{OldDataSetPath}/{Mode}"
    Classes = [Entry for Entry in os.listdir(ModePath) if os.path.isdir(os.path.join(ModePath, Entry))]
    for Class in Classes:
        ClassPath = f"{ModePath}/{Class}"
                
        Samples = [Entry for Entry in os.listdir(ClassPath) if os.path.isdir(os.path.join(ClassPath, Entry))]
        for Sample in Samples:
            SamplePath = f"{ClassPath}/{Sample}"
            ImageNames = [Entry for Entry in os.listdir(SamplePath) if Entry.endswith(".tif")]

            SavePath = f"{NewDataSetPath}/Images/{Sample}/{Class}/{Mode}"
            if not os.path.isdir(SavePath):
                os.makedirs(SavePath, exist_ok=True)

            for ImageDirectory in ImageNames:
                RandomName    = "".join(random.choices(string.digits, k=10))
                NewFilePath   = f"{SavePath}/{Sample}_{Class}_{Mode}_{RandomName}.tif"
                shutil.copy(f"{SamplePath}/{ImageDirectory}", NewFilePath)