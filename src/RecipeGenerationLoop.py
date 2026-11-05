'''
Finds a polishing recipe for a material: the shortest 3 µm and 1 µm polishing times that give a polished surface, and the
9 µm time that removes the target depth (thesis chapter 5.1, with a proper bisection).

Every trial is the full sequence on the RASI platform, followed by images of the sample (several positions):
    9 µm  for Time9um_s   removes the earlier scratches and leaves defined 9 µm scratches, so every trial starts alike
    3 µm  for the trial's 3 µm time
    1 µm  for the trial's 1 µm time
Check(images) classifies the tiles. The surface is polished if at least TargetPolished_Percent of all tiles are polished, every
image counts as polished and at most MaxUncertain_Percent of the tiles are uncertain (see ClassifyImage).

Search: the start times are tried first (doubled until they polish, up to MaxTime_s). Then the 3 µm and the 1 µm time are
bisected alternately: every stage keeps the longest time that failed and the shortest that polished, and the next trial
tries the middle, with the other stage at its shortest polishing time. So the pair of shortest times always polished
together in one trial. The search ends when both intervals are at most Resolution_s, or after MaxTrials.
Recipe:  3 µm = shortest 3 µm time + Buffer_s,  1 µm = shortest 1 µm time + Buffer_s,
         9 µm = (TargetDepth_um - removal of the 3 µm and 1 µm stages) / RemovalRate9um_um_s

From Python, e.g. in the RASI control software (see RASIExample.py):
    from RecipeGenerationLoop import RecipeGenerator, ControlVariables
    Generator = RecipeGenerator("<training run>", "22188_Sample1", ControlVariables(RemovalRate9um_um_s=..., ..., TargetDepth_um=...))
    Answer = {"Decision": "Trial", "NextTrial": Generator.FirstTrial()}
    while Answer["Decision"] == "Trial":
        Polish(Answer["NextTrial"]); Images = AcquireImages()  # RASI functions: 9, 3 and 1 µm for the given seconds
        Answer = Generator.Check(Images)                     # "Trial" (next times), "Done" (Recipe.json) or "NotPolished"

Results in src/Outputs/Recipe_<SampleName>_<date and time>/: RecipeGeneration.log, Trials.csv (one row per trial), tile maps
of every image and Recipe.json once the recipe is found.
'''
import json
import os

from dataclasses import dataclass, asdict
from datetime import datetime

import ClassifyImage as CI
import Modules.Logging as LOG

STAGES = ("3um", "1um") # the stages whose times are searched

@dataclass
class ControlVariables:
    # Inputs of the operator
    RemovalRate9um_um_s    : float = 0.0    # Material removal rates of the three polishing stages [µm/s]
    RemovalRate3um_um_s    : float = 0.0
    RemovalRate1um_um_s    : float = 0.0
    TargetDepth_um         : float = 0.0    # Material to remove per slice with the final recipe [µm]
    Time9um_s              : float = 20.0   # 9 µm polishing at the start of every trial

    # Search
    StartTime3um_s         : float = 60.0   # 3 µm and 1 µm times of the first trial; both are doubled until the surface is polished
    StartTime1um_s         : float = 60.0
    MaxTime_s              : float = 300.0  # Longest 3 µm or 1 µm time tried. If the surface is not polished then, the search stops
    Resolution_s           : float = 10.0   # The bisection of a stage stops when its interval is at most this long
    MaxTrials              : int   = 12     # Upper limit of trials (each is a full 9-3-1 µm sequence with imaging)
    Buffer_s               : float = 15.0   # Added to the shortest 3 µm and 1 µm times, so deviations between slices are still polished

    # Decision per trial
    TargetPolished_Percent : float = 90.0   # Polished if at least this share of all tiles is polished (and every image counts as polished, see ClassifyImage)
    MaxUncertain_Percent   : float = 20.0   # and at most this share of the tiles is uncertain (confidence below LowConfidence_Percent of ClassifyImage)

class RecipeGenerator():
    def __init__(self, NetworkRun: str, SampleName: str = "Sample", Controls: ControlVariables | None = None, Classifier: CI.ImageClassifier | None = None):
        # One generator per sample. The classifier (network and DINOv3) can be passed in, so it is loaded only once for several samples
        self.Controls = Controls or ControlVariables()
        Rates = [self.Controls.RemovalRate9um_um_s, self.Controls.RemovalRate3um_um_s, self.Controls.RemovalRate1um_um_s]
        if min(Rates) <= 0 or self.Controls.TargetDepth_um <= 0:
            raise ValueError("Give the removal rates of all three stages and the target depth (ControlVariables), all > 0")
        self.Classifier = Classifier or CI.ImageClassifier(NetworkRun)
        BaseDirectory   = os.path.dirname(os.path.abspath(__file__))
        self.OutputPath = f"{BaseDirectory}/Outputs/Recipe_{SampleName}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.LogFile    = LOG.NoteFile(f"{self.OutputPath}/RecipeGeneration.log")
        self.LogFile.W(f"Recipe generation for {SampleName}\nNetwork: {self.Classifier.RunPath}\n"
                       f"{json.dumps(asdict(self.Controls) | {'ScratchedTileThreshold_Percent': self.Classifier.Controls.ScratchedTileThreshold_Percent}, indent=4)}")
        with open(f"{self.OutputPath}/Trials.csv", "w") as Csv:
            Csv.write("Trial,9um_s,3um_s,1um_s,Varied,PolishedPercent,UncertainPercent,ScratchedImages,Images,Polished\n")
        self.Shortest = {Stage: None for Stage in STAGES} # shortest time that polished (with the other stage at its shortest time)
        self.Longest  = {Stage: 0.0  for Stage in STAGES} # longest time that failed
        self.Trial    = {"9um": self.Controls.Time9um_s, "3um": self.Controls.StartTime3um_s, "1um": self.Controls.StartTime1um_s}
        self.Varied   = None                              # stage bisected in this trial, None while the start times are tried
        self.Number   = 0

    def FirstTrial(self) -> dict:
        return dict(self.Trial)

    def Check(self, Images: list) -> dict:
        # Images: file paths or PIL images of the sample after the current trial. Returns the next trial or the result
        self.Number += 1
        Results = [self.Classifier.Classify(Picture, f"{self.OutputPath}/Trial{self.Number:02d}_Image{Index + 1:02d}_TileMap.png") for Index, Picture in enumerate(Images)]
        Tiles            = sum(R["Tiles"] for R in Results)
        PolishedPercent  = 100 * sum(R["Tiles"] - R["ScratchedTiles"] for R in Results) / Tiles
        UncertainPercent = 100 * sum(R["LowConfidenceTiles"] for R in Results) / Tiles
        ScratchedImages  = sum(R["Result"] == "Scratched" for R in Results)
        Polished = (PolishedPercent >= self.Controls.TargetPolished_Percent and ScratchedImages == 0 and UncertainPercent <= self.Controls.MaxUncertain_Percent)

        Times = ", ".join(f"{Stage} {Seconds:g} s" for Stage, Seconds in self.Trial.items())
        self.LogFile.W(f"\nTrial {self.Number} ({Times}): {'polished' if Polished else 'not polished'} - {PolishedPercent:.1f} % of the tiles polished, "
                       f"{UncertainPercent:.1f} % uncertain, {ScratchedImages} of {len(Results)} images scratched")
        for Index, R in enumerate(Results):
            self.LogFile.W(f"  Image {Index + 1:>2} {R['Image']:<50} {R['Result']:<9} {R['ScratchedPercent']:6.2f} % scratched tiles, {R['LowConfidenceTiles']} uncertain")
        with open(f"{self.OutputPath}/Trials.csv", "a") as Csv:
            Csv.write(f"{self.Number},{self.Trial['9um']:g},{self.Trial['3um']:g},{self.Trial['1um']:g},{self.Varied or 'start'},"
                      f"{PolishedPercent:.2f},{UncertainPercent:.2f},{ScratchedImages},{len(Results)},{Polished}\n")
        Answer = {"Trial": self.Number, "PolishedPercent": round(PolishedPercent, 2), "UncertainPercent": round(UncertainPercent, 2),
                  "ScratchedImages": ScratchedImages, "Polished": Polished, "OutputPath": self.OutputPath}

        # Update the search with the result of this trial
        if self.Varied is None:
            if not Polished:
                if max(self.Trial[Stage] for Stage in STAGES) >= self.Controls.MaxTime_s:
                    return self.Finish(Answer | {"Decision": "NotPolished", "Reason": f"not polished with 3 µm {self.Trial['3um']:g} s and 1 µm {self.Trial['1um']:g} s "
                                                                                     f"(limit {self.Controls.MaxTime_s:g} s): abrasive, cleaning or focus?"})
                # Both are doubled, but no failed time is recorded: the failure may come from only one of the stages
                for Stage in STAGES:
                    self.Trial[Stage] = min(2 * self.Trial[Stage], self.Controls.MaxTime_s)
                return self.Next(Answer, "start times not polished, both doubled")
            self.Shortest = {Stage: self.Trial[Stage] for Stage in STAGES}
        elif Polished:
            self.Shortest[self.Varied] = self.Trial[self.Varied]
        else:
            self.Longest[self.Varied] = self.Trial[self.Varied]

        # Next stage to bisect: alternately, as long as its interval is longer than Resolution_s
        Open = [Stage for Stage in STAGES if self.Shortest[Stage] - self.Longest[Stage] > self.Controls.Resolution_s]
        if not Open or self.Number >= self.Controls.MaxTrials:
            return self.Finish(Answer | {"Decision": "Done", "Reason": "intervals of both stages within the resolution" if not Open else f"{self.Controls.MaxTrials} trials reached"})
        Stage = next((Other for Other in Open if Other != self.Varied), Open[0])
        self.Varied = Stage
        self.Trial  = {"9um": self.Controls.Time9um_s} | {Other: self.Shortest[Other] for Other in STAGES}
        self.Trial[Stage] = float(round((self.Shortest[Stage] + self.Longest[Stage]) / 2))
        return self.Next(Answer, f"{Stage}: shortest polished {self.Shortest[Stage]:g} s, longest failed {self.Longest[Stage]:g} s")

    def Next(self, Answer: dict, Reason: str) -> dict:
        self.LogFile.W(f"  Next trial: {', '.join(f'{Stage} {Seconds:g} s' for Stage, Seconds in self.Trial.items())} ({Reason})")
        return Answer | {"Decision": "Trial", "Reason": Reason, "NextTrial": dict(self.Trial)}

    def Finish(self, Answer: dict) -> dict:
        if Answer["Decision"] == "Done":
            C = self.Controls
            Times   = {Stage: self.Shortest[Stage] + C.Buffer_s for Stage in STAGES}
            Removed = {"3um": C.RemovalRate3um_um_s * Times["3um"], "1um": C.RemovalRate1um_um_s * Times["1um"]}
            Removed["9um"] = C.TargetDepth_um - Removed["3um"] - Removed["1um"]
            Times["9um"]   = Removed["9um"] / C.RemovalRate9um_um_s
            Recipe = {"Polishing_s": {Stage: round(Times[Stage], 1) for Stage in ("9um", "3um", "1um")},
                      "Removal_um": {Stage: round(Removed[Stage], 3) for Stage in ("9um", "3um", "1um")}, "TargetDepth_um": C.TargetDepth_um,
                      "ShortestPolished_s": self.Shortest, "Buffer_s": C.Buffer_s, "Trials": self.Number, "Network": self.Classifier.RunPath, "Settings": asdict(C)}
            if Times["9um"] < C.Time9um_s:
                Recipe["Warning"] = f"The 9 µm time for the target depth is shorter than the {C.Time9um_s:g} s of the trials: the grinding scratches may remain"
            with open(f"{self.OutputPath}/Recipe.json", "w") as RecipeFile:
                json.dump(Recipe, RecipeFile, indent=4)
            Answer |= {"Recipe": Recipe}
            Answer["Reason"] += (f". Recipe: 9 µm {Times['9um']:.0f} s, 3 µm {Times['3um']:g} s, 1 µm {Times['1um']:g} s "
                                 f"(shortest polished 3 µm {self.Shortest['3um']:g} s and 1 µm {self.Shortest['1um']:g} s, + {C.Buffer_s:g} s each)"
                                 + (f". WARNING: {Recipe['Warning']}" if "Warning" in Recipe else ""))
        self.LogFile.W(f"\n{Answer['Decision']}: {Answer['Reason']}")
        self.LogFile.Close()
        return Answer
