'''
Example of how the RASI control software uses RecipeGenerationLoop.py to find a polishing recipe for a material.
RASI is controlled from Python (robot, polisher and the ZEN microscope jobs, e.g. utils_adaptive.py of the thesis code). The
calls to the machine are collected in the class RASI below, so they can be replaced by a simulation:

    python src/RASIExample.py --network <RunName> --rates <9 µm> <3 µm> <1 µm> --depth <µm> --polished <folder> --scratched <folder>
                              [--minimum <3 µm s> <1 µm s>] [--sample <SampleName>]

The simulation knows the "true" shortest 3 µm and 1 µm times (--minimum). A trial with at least these times returns the images of
the polished folder, any other trial those of the scratched folder; the images themselves are classified by the network. To use
it on the machine, fill in the methods of RASI with the calls of the control software and create RASI() instead of SimulatedRASI(...).
'''
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # makes ClassifyImage, Modules etc. importable from any working directory
import ClassifyImage as CI
from RecipeGenerationLoop import RecipeGenerator, ControlVariables

class RASI():
    # The machine. Replace the bodies with the calls of the RASI control software
    def Polish(self, Trial: dict):
        # Trial: {"9um": s, "3um": s, "1um": s}. Polish the stages in this order, cleaning the sample after each, e.g.
        # command_robot("121"); start_polisher_adaptive(<recipe for the three times>, recipes)
        raise NotImplementedError("Fill in the polishing calls of the RASI control software")

    def AcquireImages(self) -> list[str]:
        # e.g. command_robot("112"); start_zen_job_adapt("ASQA_imaging_job"); listen_to_microscope_adapt(5042)
        # Returns the image files of this trial only (the ZEN job writes them to its output folder)
        raise NotImplementedError("Fill in the imaging calls of the RASI control software")

class SimulatedRASI(RASI):
    # Polished images if the trial's 3 µm and 1 µm times reach the given minimum, otherwise scratched images
    def __init__(self, Minimum: dict, PolishedFolder: str, ScratchedFolder: str):
        self.Minimum, self.Polished, self.Scratched = Minimum, CI.FindImages(PolishedFolder), CI.FindImages(ScratchedFolder)

    def Polish(self, Trial: dict):
        self.Trial = Trial
        print(f"  [simulation] polish {', '.join(f'{Stage} {Seconds:g} s' for Stage, Seconds in Trial.items())}")

    def AcquireImages(self) -> list[str]:
        return self.Polished if all(self.Trial[Stage] >= Seconds for Stage, Seconds in self.Minimum.items()) else self.Scratched

def GenerateRecipe(Machine: RASI, Generator: RecipeGenerator, ArchiveImages: bool) -> dict:
    # Trials until RecipeGenerator has found the recipe (or stops)
    Trial = Generator.FirstTrial()
    while True:
        Machine.Polish(Trial)
        Images = Machine.AcquireImages()
        Answer = Generator.Check(Images)
        print(f"Trial {Answer['Trial']}: {'polished' if Answer['Polished'] else 'not polished'} ({Answer['PolishedPercent']:.1f} % of the tiles) "
              f"-> {Answer['Decision']}: {Answer['Reason']}")
        if ArchiveImages:
            # The next trial must only see its own images: move them next to the results
            Archive = f"{Generator.OutputPath}/Images/Trial{Answer['Trial']:02d}"
            os.makedirs(Archive, exist_ok=True)
            for File in Images:
                shutil.move(File, Archive)
        if Answer["Decision"] != "Trial":
            return Answer
        Trial = Answer["NextTrial"]

def Main():
    Parser = argparse.ArgumentParser(description="RASI example: find a polishing recipe (simulated machine).")
    Parser.add_argument("--network", required=True, help="Training run in src/Outputs/")
    Parser.add_argument("--sample", default="Sample", help="Sample name for the output folder")
    Parser.add_argument("--rates", type=float, nargs=3, required=True, metavar=("R9", "R3", "R1"), help="Removal rates of the 9, 3 and 1 µm stages [µm/s]")
    Parser.add_argument("--depth", type=float, required=True, help="Target removal depth per slice [µm]")
    Parser.add_argument("--polished", required=True, help="Simulation: folder with images of a polished surface")
    Parser.add_argument("--scratched", required=True, help="Simulation: folder with images of a scratched surface")
    Parser.add_argument("--minimum", type=float, nargs=2, default=[50, 70], metavar=("T3", "T1"), help="Simulation: true shortest 3 and 1 µm times [s]")
    Arguments = Parser.parse_args()

    Controls  = ControlVariables(RemovalRate9um_um_s=Arguments.rates[0], RemovalRate3um_um_s=Arguments.rates[1],
                                 RemovalRate1um_um_s=Arguments.rates[2], TargetDepth_um=Arguments.depth)
    Machine   = SimulatedRASI({"3um": Arguments.minimum[0], "1um": Arguments.minimum[1]}, Arguments.polished, Arguments.scratched)  # on the machine: RASI()
    Generator = RecipeGenerator(Arguments.network, Arguments.sample, Controls)
    Answer    = GenerateRecipe(Machine, Generator, ArchiveImages=False)   # on the machine: True. The simulated images are reused

    if Answer["Decision"] == "Done":
        print(f"Recipe {Answer['Recipe']['Polishing_s']} s: {Generator.OutputPath}/Recipe.json")
    else:
        print(f"No recipe ({Answer['Reason']}). Stop and keep the sample for inspection: {Generator.OutputPath}")
    # For a further sample, reuse the loaded network: RecipeGenerator(Arguments.network, "<next sample>", Controls, Classifier=Generator.Classifier)

if __name__ == "__main__":
    Main()
