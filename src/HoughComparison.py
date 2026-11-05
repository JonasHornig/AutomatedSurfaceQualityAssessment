'''
Conventional comparison method of the thesis (Hornig, chapters 2, 3.3 and 4.2): straight line detection with a Hough transform on whole
micrographs. A scratch is a long straight edge. After edge detection, every edge pixel votes for all straight lines through it; each peak
of this Hough accumulator is a line, with the number of edge pixels on it as its strength. The Hough value of an image is the mean
strength of its strongest lines, and the image is scratched from a fixed threshold on.
It does not use the network, so it is an independent and interpretable check of the network's classification.

Changes compared with the thesis implementation:
  - Python only (scikit-image) instead of an ImageJ macro with fixed paths, and a vectorised Hough transform (0.5° and 1 px steps)
  - Canny on the contrast stretched grey image instead of on a binarised image, because the binarisation also removed faint scratches
    (thesis 6.1.2). The contrast is stretched over the matrix only: dark graphite would otherwise use up the contrast range, and the
    faint scratches next to it would give no edges
  - The mean of the 10 strongest lines instead of the single strongest one: a polished micrograph often has one or two isolated scratches,
    which give one strong line, whereas a scratched surface gives many
  - One fixed threshold for all materials instead of one read from a histogram

Tuned for these materials: the matrix stretch, the number of lines and the threshold were chosen on 60 whole micrographs of the three
training materials (10 per material and state), on which they classify all 60 correctly. They are not validated on other micrographs,
and every new material, especially one with another microstructure, would need this tuning again. The network needs no such handcrafting.

Command line:
    python src/HoughComparison.py <image or folder> [...] [--threshold <Hough value>]
Images whose file name contains "polished" or "scratched" are scored against that label, per material (the file name up to the first "_").
Results in src/Outputs/Hough_<date and time>/: HoughComparison.log, Hough.csv, the Hough values of all images (HoughValues.png) and a
Hough diagram (image, detected edges with the strongest lines, and Hough space) of every image.
'''
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from datetime import datetime
from PIL import Image
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.transform import hough_line, hough_line_peaks
from tqdm import tqdm

import Modules.Logging as LOG

@dataclass
class ControlVariables:
    Threshold                : float = 300   # Hough value (mean strength of the strongest lines) from which an image is scratched
    NumberOfLines            : int   = 10    # Strongest lines averaged into the Hough value
    MinLineDistance_px       : int   = 25    # Peaks closer than this and than MinLineAngle count as one line (e.g. both edges of one scratch)
    MinLineAngle_deg         : float = 5.0

    # Edge detection and Hough transform
    MatrixFraction           : float = 0.75  # Pixels darker than this fraction of the median grey value (graphite) are left out of the contrast stretch
    Sigma                    : float = 3.0   # Gaussian smoothing of the Canny edge detector [px]
    LowThreshold             : float = 0.04  # Canny hysteresis thresholds on the gradient of the contrast stretched image (0 to 1)
    HighThreshold            : float = 0.10
    MinEdgeSize_px           : int   = 50    # Edges with at most this many pixels are removed (dirt, small microstructure features)
    ThetaStep_deg            : float = 0.5   # Angle resolution of the Hough space

def HoughSpace(Picture: Image.Image, Controls: ControlVariables):
    # Edges, Hough accumulator [distances, angles], angles [rad] and distances [px] of an image
    Gray = np.asarray(Picture.convert("L"), dtype=float)
    # Contrast stretch over the matrix with 1 % saturated pixels, like "Enhance Contrast" in ImageJ; the graphite becomes black
    Low, High = np.percentile(Gray[Gray > Controls.MatrixFraction * np.median(Gray)], (0.5, 99.5))
    Gray = np.clip((Gray - Low) / max(High - Low, 1e-6), 0, 1)
    Edges = canny(Gray, sigma=Controls.Sigma, low_threshold=Controls.LowThreshold, high_threshold=Controls.HighThreshold)
    Edges = remove_small_objects(Edges, max_size=Controls.MinEdgeSize_px, connectivity=2)
    Accumulator, Angles, Distances = hough_line(Edges, theta=np.deg2rad(np.arange(-90, 90, Controls.ThetaStep_deg)))
    return Edges, Accumulator, Angles, Distances

def StrongestLines(Accumulator, Angles, Distances, Controls: ControlVariables):
    # Strengths, angles [rad] and distances [px] of the strongest separate lines
    return hough_line_peaks(Accumulator, Angles, Distances, num_peaks=Controls.NumberOfLines, threshold=1, min_distance=Controls.MinLineDistance_px,
                            min_angle=round(Controls.MinLineAngle_deg / Controls.ThetaStep_deg))

def PlotHoughDiagram(Picture: Image.Image, Edges, Accumulator, Angles, Distances, Lines, Value: float, Controls: ControlVariables, Title: str, Path: str):
    # Image, detected edges with the strongest lines, and Hough space with their peaks, as in figure 25 of the thesis
    Figure, Axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    Axes[0].imshow(Picture)
    Axes[0].set_title("Image")
    Axes[1].imshow(Picture.convert("L"), cmap="gray")
    Axes[1].imshow(np.ma.masked_where(~Edges, Edges), cmap="autumn", interpolation="none")
    # Lines shaded by rank: the strongest dark blue and drawn on top, the weakest light blue
    Order   = np.argsort(Lines[0])[::-1]
    Colours = plt.cm.Blues(np.linspace(1.0, 0.35, len(Order)))
    X = np.array([0, Picture.width - 1])
    for Rank, Index in enumerate(Order):                         # x cos(theta) + y sin(theta) = rho
        Theta, Rho, Style = Lines[1][Index], Lines[2][Index], {"color": Colours[Rank], "linewidth": 1.4 if Rank == 0 else 0.8, "zorder": 3 + len(Order) - Rank}
        if abs(np.sin(Theta)) > 1e-3:
            Axes[1].plot(X, (Rho - X * np.cos(Theta)) / np.sin(Theta), **Style)
        else:
            Axes[1].axvline(Rho, **Style)
    Axes[1].set_xlim(0, Picture.width - 1); Axes[1].set_ylim(Picture.height - 1, 0)
    Axes[1].set_title(f"Edges ({int(Edges.sum())} px) and the {len(Order)} strongest lines\n(darker = stronger, strongest {Lines[0][Order[0]]} px)")
    for Axis in Axes[:2]:
        Axis.axis("off")
    Shown = Axes[2].imshow(Accumulator, cmap="Greys", aspect="auto", origin="lower",
                           extent=(np.rad2deg(Angles[0]), np.rad2deg(Angles[-1]), Distances[0], Distances[-1]))
    Axes[2].scatter(np.rad2deg(Lines[1][Order]), Lines[2][Order], s=60, facecolors="none", edgecolors=Colours, linewidths=1.4)
    Figure.colorbar(Shown, ax=Axes[2], label="Edge pixels on the line")
    Axes[2].set_xlabel("θ [°]"); Axes[2].set_ylabel("ρ [px]")
    Axes[2].set_title(f"Hough value {Value:.0f} (threshold {Controls.Threshold:g})")
    Figure.suptitle(Title, fontsize=13)
    Figure.tight_layout()
    Figure.savefig(Path, dpi=120)
    plt.close(Figure)

def PlotHoughValues(Rows: list[dict], Threshold: float, Path: str):
    # Hough value of every image, one column per material, coloured by the label of the image
    Materials = list(dict.fromkeys(R["Material"] for R in Rows))
    Colours   = {"Polished": "#2a78d6", "Scratched": "#eb6834", "Unknown": "#6b7280"}
    Figure, Axis = plt.subplots(figsize=(1.8 + 1.6 * len(Materials), 4.5))
    Random = np.random.default_rng(0)                            # horizontal jitter, so equal values stay visible
    for State, Colour in Colours.items():
        Labelled = False
        for k, Material in enumerate(Materials):
            Values = [R["HoughValue"] for R in Rows if R["Material"] == Material and R["Label"] == State]
            if Values:
                Offset = {"Polished": -0.15, "Scratched": 0.15}.get(State, 0)
                Axis.scatter(k + Offset + Random.uniform(-0.08, 0.08, len(Values)), Values, s=28, color=Colour, edgecolors="white", linewidths=0.5,
                             label=None if Labelled else State, zorder=3)
                Labelled = True
    Axis.axhline(Threshold, color="#1f1f1f", linestyle="--", linewidth=1, label=f"Threshold {Threshold:g}", zorder=2)
    Axis.set_xticks(range(len(Materials)), Materials)
    Axis.set_xlim(-0.6, len(Materials) - 0.4)
    Axis.set_ylabel("Hough value (mean edge pixels on the strongest lines)")
    Axis.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    Axis.spines[["top", "right"]].set_visible(False)
    Figure.tight_layout()
    Figure.savefig(Path, dpi=200, bbox_inches="tight")
    plt.close(Figure)

def FindImages(Path: str) -> list[str]:
    if os.path.isfile(Path):
        return [Path]
    return [os.path.join(Path, Entry) for Entry in sorted(os.listdir(Path)) if Entry.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]

def Main():
    Defaults = ControlVariables()
    Parser   = argparse.ArgumentParser(description="Hough transform classification (thesis method) of whole micrographs.")
    Parser.add_argument("Images", nargs="+", help="Images or folders of images")
    Parser.add_argument("--threshold", type=float, default=Defaults.Threshold, help=f"Hough value from which an image is scratched (default {Defaults.Threshold:g})")
    Arguments = Parser.parse_args()
    Controls  = ControlVariables(Threshold=Arguments.threshold)

    OutputPath = f"{os.path.dirname(os.path.abspath(__file__))}/Outputs/Hough_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"{OutputPath}/HoughDiagrams")
    LogFile = LOG.NoteFile(f"{OutputPath}/HoughComparison.log")
    LogFile.W(f"Hough transform classification of whole micrographs\n{json.dumps(asdict(Controls), indent=4)}\n")

    Rows = []
    for File in tqdm([File for Path in Arguments.Images for File in FindImages(Path)], desc="Hough transform", unit=" images"):
        with Image.open(File) as Opened:
            Picture = Opened.convert("RGB")
        Name  = os.path.splitext(os.path.basename(File))[0]
        Edges, Accumulator, Angles, Distances = HoughSpace(Picture, Controls)
        Lines = StrongestLines(Accumulator, Angles, Distances, Controls)
        Value = float(np.mean(Lines[0]))
        Label = next((State for State in ("Polished", "Scratched") if State.lower() in Name.lower()), "Unknown")
        Rows.append({"Image": Name, "Material": Name.split("_")[0], "Label": Label, "HoughValue": Value,
                     "Hough": "Scratched" if Value >= Controls.Threshold else "Polished"})
        PlotHoughDiagram(Picture, Edges, Accumulator, Angles, Distances, Lines, Value, Controls, f"{Name}: {Rows[-1]['Hough']}", f"{OutputPath}/HoughDiagrams/{Name}.png")

    with open(f"{OutputPath}/Hough.csv", "w") as Csv:
        Csv.write("Image,Material,Label,HoughValue,Hough\n")
        Csv.writelines(f"{R['Image']},{R['Material']},{R['Label']},{R['HoughValue']:.1f},{R['Hough']}\n" for R in Rows)
    PlotHoughValues(Rows, Controls.Threshold, f"{OutputPath}/HoughValues.png")

    # Correct images per material and label, for images whose file name gives the label
    Labelled = [R for R in Rows if R["Label"] != "Unknown"]
    if Labelled:
        LogFile.W(f"Correct images (scratched from a Hough value of {Controls.Threshold:g} on):")
        for Material in list(dict.fromkeys(R["Material"] for R in Labelled)) + ["All"]:
            Part = [R for R in Labelled if Material in (R["Material"], "All")]
            Line = f"  {Material:<10}" + "".join(f"  {State.lower()} {sum(R['Hough'] == State for R in Part if R['Label'] == State)} of {sum(R['Label'] == State for R in Part)}"
                                             for State in ("Polished", "Scratched"))
            LogFile.W(Line); print(Line)
    LogFile.Close()
    print(f"Results: {OutputPath}")

if __name__ == "__main__":
    Main()
