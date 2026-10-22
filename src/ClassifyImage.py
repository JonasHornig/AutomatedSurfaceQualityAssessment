'''
Classifies complete microscope images (e.g. 2464 x 2056 pixel RASI images) as polished or scratched.
Every image is cut into 224 x 224 pixel tiles (99 tiles for a 2464 x 2056 image), the surface state of every tile is
classified with a trained network, and the image counts as scratched if the share of scratched tiles exceeds a threshold.

Command line (prints one line per image, or JSON with --json):
    python src/ClassifyImage.py <image or folder> [...] --network <training run in src/Outputs/> [--threshold 20] [--json] [--no-tilemap]

From Python, e.g. in the RASI loop (the networks are loaded once, then every image takes a few seconds):
    from ClassifyImage import ImageClassifier
    Classifier = ImageClassifier("<training run in src/Outputs/>")
    Result     = Classifier.Classify("Image.tif")   # {"Result": "Polished", "ScratchedPercent": 1.01, ...}
'''
import argparse
import json
import os
import time
import torch

from dataclasses import dataclass
from datetime import datetime
from PIL import Image, ImageDraw

import Modules.NeuralNetwork as NN
import Modules.NetworkTesting as TEST
import Modules.Logging as LOG

@dataclass
class ControlVariables:
    # Defaults, used when the script is run without command line arguments (e.g. from the editor) and for the command line options
    ImagePath  : str = "" # A single image or a folder of images (.tif, .tiff, .png, .jpg), relative to src/ or absolute
    NetworkRun : str = "" # Output folder of a training run in src/Outputs/ (contains Settings.json, TrainedProtoNet.pth, PrototypeEmbedding.pth, Labelmap.json)

    # Decision
    ScratchedTileThreshold_Percent : float = 20.0        # An image is scratched if more than this share of its tiles is scratched (thesis: polished images had at most 5 %; provisional value)
    ScratchedState                 : str   = "Scratched" # Name of the surface state that counts as scratched
    LowConfidence_Percent          : float = 90.0        # Tiles with a lower surface state confidence are counted as uncertain

    # Image handling
    TileSize     : int  = 224  # Tiles are cut at full resolution, as for the training images. The grid is centred, the remaining border is not used
    MaxBatchSize : int  = 16   # Maximum number of tiles passed through DINOv3 at once
    SaveTileMaps : bool = True # Saves a copy of every image with the surface state of each tile marked

class SilentLog():
    # Stands in for a log file when the classifier is used without one
    def W(self, Text, NewLine: bool = True):
        pass

class ImageClassifier():
    def __init__(self, NetworkRun: str, Controls: ControlVariables | None = None, LogFile=None):
        # Loads the trained network and the matching DINOv3 model once
        self.Controls = Controls or ControlVariables()
        self.LogFile  = LogFile or SilentLog()
        BaseDirectory = os.path.dirname(os.path.abspath(__file__))
        if not NetworkRun:
            raise ValueError("Give the output folder of a training run in src/Outputs/ (NetworkRun / --network)")
        self.RunPath  = NetworkRun if os.path.isabs(NetworkRun) else f"{BaseDirectory}/Outputs/{NetworkRun}"

        Start = time.time()
        self.Network, Settings, self.LookUpTable = NN.ProtoNet.LoadNetwork(self.LogFile, self.RunPath)
        self.FeatureExtractor = Settings["FeatureExtractor"]
        if self.FeatureExtractor not in NN.DINOV3_MODELS:
            raise ValueError(f"The network was trained on {self.FeatureExtractor} features, which can't be extracted with this code. Use a network trained on {list(NN.DINOV3_MODELS)} features.")
        self.DinoModel       = NN.DINOv3Model(self.FeatureExtractor, f"{BaseDirectory}/Models")
        self.PrototypeStates = [TEST.SurfaceState(self.LookUpTable[Label]) for Label in range(len(self.LookUpTable))]
        self.LoadingSeconds  = time.time() - Start

    def Classify(self, Picture: str | Image.Image, TileMapPath: str | None = None) -> dict:
        '''
        Classifies one image, given as a file path or a PIL image. Returns the result and the factors it is based on:
          Result                     "Polished" or "Scratched"
          ScratchedPercent           share of scratched tiles, compared with ThresholdPercent
          ScratchedTiles, Tiles      number of scratched tiles and of all tiles
          MeanStateConfidencePercent mean surface state confidence of the tiles (summed probability of all prototypes with the predicted state)
          LowConfidenceTiles         number of tiles with a surface state confidence below LowConfidence_Percent
          TilesToChangeResult        how many tiles would have to change their surface state to change the result (robustness of the result)
          TileStates, TileConfidencesPercent, TileBoxes   per tile results, boxes as (left, top, right, bottom) in pixels
        '''
        Start = time.time()
        Name  = Picture if isinstance(Picture, str) else "image"
        if isinstance(Picture, str):
            with Image.open(Picture) as OpenedImage:
                Picture = OpenedImage.convert("RGB")
        else:
            Picture = Picture.convert("RGB")

        Embeddings, Boxes, _ = self.EmbedTiles(Picture, Name)
        Probabilities     = torch.softmax(-NN.EuclideanDistance(Embeddings, self.Network.Prototypes), dim=1)
        States    = [self.PrototypeStates[Label] for Label in Probabilities.argmax(1).tolist()]
        Confidences = [100 * sum(P for P, PrototypeState in zip(Row, self.PrototypeStates) if PrototypeState == State) for Row, State in zip(Probabilities.tolist(), States)]

        Threshold        = self.Controls.ScratchedTileThreshold_Percent
        ScratchedTiles   = sum(State == self.Controls.ScratchedState for State in States)
        ScratchedPercent = 100 * ScratchedTiles / len(States)
        Result           = "Scratched" if ScratchedPercent > Threshold else "Polished"
        MaximumPolished  = int(Threshold * len(States) / 100 + 1e-9) # largest number of scratched tiles that still counts as polished
        TilesToChange    = ScratchedTiles - MaximumPolished if Result == "Scratched" else MaximumPolished + 1 - ScratchedTiles

        if TileMapPath:
            SaveTileMap(Picture, Boxes, States, self.Controls.ScratchedState, TileMapPath)
        return {"Image": os.path.basename(Name), "Result": Result, "ScratchedPercent": round(ScratchedPercent, 2), "ThresholdPercent": Threshold,
                "ScratchedTiles": ScratchedTiles, "Tiles": len(States), "MeanStateConfidencePercent": round(sum(Confidences) / len(Confidences), 2),
                "LowConfidenceTiles": sum(Confidence < self.Controls.LowConfidence_Percent for Confidence in Confidences),
                "TilesToChangeResult": TilesToChange, "Seconds": round(time.time() - Start, 2), "Network": self.RunPath, "FeatureExtractor": self.FeatureExtractor,
                "TileStates": States, "TileConfidencesPercent": [round(Confidence, 2) for Confidence in Confidences], "TileBoxes": Boxes}

    def EmbedTiles(self, Picture: Image.Image, Name: str = "image") -> tuple[torch.Tensor, list[tuple[int, int, int, int]], torch.Tensor]:
        # Embeddings [tiles, embedding dimension], boxes and DINO features [tiles, feature dimension] of the tiles of an RGB image
        Tiles, Boxes = CutTiles(Picture, self.Controls.TileSize)
        if not Tiles:
            raise ValueError(f"{Name} ({Picture.size[0]} x {Picture.size[1]}) is smaller than one tile ({self.Controls.TileSize} x {self.Controls.TileSize})")
        TileData = {"Images": Tiles, "Names": [f"Tile_{Index}" for Index in range(len(Tiles))]}
        self.DinoModel.ApplyNetwork(TileData, self.LogFile, MaxBatchSize=self.Controls.MaxBatchSize)
        with torch.no_grad():
            return self.Network.Encoder(TileData["DinoFeatures"]), Boxes, TileData["DinoFeatures"]

def FindImages(Path: str) -> list[str]:
    if os.path.isfile(Path):
        return [Path]
    if os.path.isdir(Path):
        return [os.path.join(Path, Entry) for Entry in sorted(os.listdir(Path)) if Entry.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    raise FileNotFoundError(f"{Path} is neither an image nor a folder")

def CutTiles(Picture: Image.Image, TileSize: int) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    # Centred grid of full resolution tiles, returned with their boxes (left, top, right, bottom) in the image
    Width, Height    = Picture.size
    Columns, Rows    = Width // TileSize, Height // TileSize
    OffsetX, OffsetY = (Width - Columns * TileSize) // 2, (Height - Rows * TileSize) // 2
    Boxes = [(OffsetX + Column * TileSize, OffsetY + Row * TileSize, OffsetX + (Column + 1) * TileSize, OffsetY + (Row + 1) * TileSize)
             for Row in range(Rows) for Column in range(Columns)]
    return [Picture.crop(Box) for Box in Boxes], Boxes

def SaveTileMap(Picture: Image.Image, Boxes, States, ScratchedState: str, Path: str):
    # Scratched tiles in orange, polished tiles in blue (colour blind safe pair), downscaled to half size
    Overlay = Picture.convert("RGBA")
    Layer   = Overlay.copy(); Layer.putalpha(0)
    Draw    = ImageDraw.Draw(Layer)
    for Box, State in zip(Boxes, States):
        Colour = (235, 104, 52) if State == ScratchedState else (42, 120, 214)
        Draw.rectangle(Box, fill=Colour + (60,), outline=Colour + (255,), width=4)
    Result = Image.alpha_composite(Overlay, Layer).convert("RGB")
    Result.resize((Result.width // 2, Result.height // 2)).save(Path)

def Main():
    Defaults = ControlVariables()
    Parser   = argparse.ArgumentParser(description="Classify microscope images as polished or scratched.")
    Parser.add_argument("Images", nargs="*", default=[Defaults.ImagePath] if Defaults.ImagePath else [], help="Images or folders of images")
    Parser.add_argument("--network", default=Defaults.NetworkRun, help="Output folder of a training run in src/Outputs/ (or an absolute path)")
    Parser.add_argument("--threshold", type=float, default=Defaults.ScratchedTileThreshold_Percent, help="Scratched if more than this percentage of tiles is scratched")
    Parser.add_argument("--output", default="", help="Output folder (default: src/Outputs/Classification_<date and time>)")
    Parser.add_argument("--json", action="store_true", help="Print the results as JSON (without the per tile lists)")
    Parser.add_argument("--no-tilemap", action="store_true", help="Don't save the tile maps")
    Arguments = Parser.parse_args()
    if not Arguments.Images:
        Parser.error("No image given (command line or ControlVariables.ImagePath)")

    Controls = ControlVariables(ScratchedTileThreshold_Percent=Arguments.threshold, SaveTileMaps=not Arguments.no_tilemap)
    BaseDirectory = os.path.dirname(os.path.abspath(__file__))
    OutputPath    = Arguments.output or f"{BaseDirectory}/Outputs/Classification_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LogFile       = LOG.NoteFile(f"{OutputPath}/Classification.log")
    LogFile.W("\n*********************************\n*  Image surface classification  *\n*********************************")

    Classifier = ImageClassifier(Arguments.network, Controls, LogFile)
    LogFile.W(f"Network: {Classifier.RunPath} ({Classifier.FeatureExtractor} features on the {Classifier.DinoModel.Device.type.upper()}, loaded in {Classifier.LoadingSeconds:.1f} s)")
    LogFile.W(f"Decision: scratched if more than {Controls.ScratchedTileThreshold_Percent} % of the tiles are {Controls.ScratchedState}")

    Results = []
    with open(f"{OutputPath}/Classification.csv", "w") as ImageCsv, open(f"{OutputPath}/Tiles.csv", "w") as TileCsv:
        ImageCsv.write("Image,Result,ScratchedPercent,ThresholdPercent,ScratchedTiles,Tiles,MeanStateConfidencePercent,LowConfidenceTiles,TilesToChangeResult,Seconds\n")
        TileCsv.write("Image,Tile,Left,Top,State,StateConfidencePercent\n")
        for ImageFile in [File for Path in Arguments.Images for File in FindImages(Path)]:
            Name    = os.path.splitext(os.path.basename(ImageFile))[0]
            TileMap = f"{OutputPath}/{Name}_TileMap.png" if Controls.SaveTileMaps else None
            R       = Classifier.Classify(ImageFile, TileMap)
            Results.append(R)
            ImageCsv.write(",".join(str(R[Key]) for Key in ["Image", "Result", "ScratchedPercent", "ThresholdPercent", "ScratchedTiles", "Tiles", "MeanStateConfidencePercent", "LowConfidenceTiles", "TilesToChangeResult", "Seconds"]) + "\n")
            for Index, (Box, State, Confidence) in enumerate(zip(R["TileBoxes"], R["TileStates"], R["TileConfidencesPercent"])):
                TileCsv.write(f"{R['Image']},{Index},{Box[0]},{Box[1]},{State},{Confidence}\n")
            Summary = (f"{R['Image']}: {R['Result']} - {R['ScratchedTiles']} of {R['Tiles']} tiles scratched ({R['ScratchedPercent']} %, threshold {R['ThresholdPercent']} %), "
                       f"mean confidence {R['MeanStateConfidencePercent']} %, uncertain tiles: {R['LowConfidenceTiles']}, tiles from changing the result: {R['TilesToChangeResult']}, {R['Seconds']} s")
            LogFile.W(Summary)
            if not Arguments.json:
                print(Summary)
    if Arguments.json:
        PerTile = {"TileStates", "TileConfidencesPercent", "TileBoxes"}
        print(json.dumps([{Key: Value for Key, Value in R.items() if Key not in PerTile} for R in Results], indent=2))
    LogFile.Close()

if __name__ == "__main__":
    Main()
