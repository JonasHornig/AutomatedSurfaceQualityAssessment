# Automated Surface Quality Assessment (ASQA)

Classification of metallographic surfaces by the quality of their polishing finish, developed for the
Robot-Assisted Automated Serial-Sectioning and Imaging (RASI) platform at the
Bundesanstalt für Materialforschung und -prüfung (BAM), Berlin.

Microscope image tiles are classified into *polished* or *scratched* for each material. The
classifier is a [prototypical network](https://arxiv.org/abs/1703.05175) (Snell et al., 2017) that
works on frozen image features of a pretrained [DINOv3](https://github.com/facebookresearch/dinov3)
vision transformer.

## Method

1. **Feature extraction** (`src/PreprocessImages.py`): each 224 × 224 RGB tile is passed through a frozen
   DINOv3 model. Its class token is stored as the feature vector of the tile in an HDF5 file.
2. **Embedding network** (`src/Modules/NeuralNetwork.py`): a custom NN maps the feature vector
   (768 dimensions for ViT-B/16) through `NumberOfHiddenLayers` hidden blocks (Linear, BatchNorm, LeakyReLU,
   2 by default) and a linear output layer into a low dimensional embedding space (8 dimensions by default).
   The layers are counted as follows:

   | Layer          | In this code                                                                                      |
   |----------------|---------------------------------------------------------------------------------------------------|
   | Input layer    | The DINOv3 feature vector itself (768 values for ViT-B/16). It has no weights.                    |
   | Hidden layer   | One block of Linear, BatchNorm and LeakyReLU. Its output vector is the hidden representation. `NumberOfHiddenLayers` counts these blocks. |
   | Output layer   | A plain Linear layer into the embedding space, without an activation.                            |

   `NumberOfHiddenLayers = H` therefore gives H hidden blocks and one output layer, i.e. H + 1 Linear layers.
   With `NumberOfHiddenLayers = 0` the network is a single linear transformation of the feature vector.
3. **Episodic training** (`src/TrainNetwork.py`): every episode draws `N_C` classes with `N_S` support and
   `N_Q` query images each. The prototype of a class is the mean embedding of its support images, and the
   loss is the cross entropy over the negative squared Euclidean distances of the queries to the prototypes.
4. **Validation**: a fixed, seeded fraction of the training images of every class (`ValidationFraction`)
   is held back and never used as support or query image. It is used to monitor the training.
5. **Testing**: after training, the final prototypes are computed from all training images (without the
   validation images), and each test image is assigned to the class of the nearest prototype.

By default (`ClassGrouping = "MaterialAndState"`) a class is the combination of material and surface state,
e.g. `18201-GJSCastIron_Polished`, so every surface state is represented by one prototype per material. For the surface
quality only the surface state of the predicted class matters. Its confidence is the summed probability of all
prototypes with that surface state. `ClassGrouping = "State"` instead pools the materials into one class per
surface state.

## Results

`CompleteDataSet`, default settings of `TrainNetwork.py` (2000 episodes, learning rate `1e-3` with a cosine
schedule, `N_C = 6`, `N_S = 5`, `N_Q = 10`, 2 hidden layers, 8 embedding dimensions), mean ± standard deviation
over five seeds, 15,000 test images:

| Features           | Validation accuracy | Test accuracy, all classes | Surface state (polished vs. scratched) | Classifying one RASI image (99 tiles, laptop CPU) |
|--------------------|---------------------|----------------------------|----------------------------------------|---------------------------------------------------|
| DINOv3 ViT-B/16    | 97.8 ± 0.2 %        | 98.33 ± 0.06 %             | 99.86 ± 0.01 %                         | about 20 s                                        |
| DINOv3 ViT-H+/16   | 97.9 ± 0.2 %        | 98.56 ± 0.05 %             | 99.86 ± 0.02 %                         | about 160 s                                       |

Both models reach the same surface state accuracy (19 to 26 of 15,000 test images with a wrong surface state),
so ViT-B/16 is the recommended feature extractor. Most remaining errors are confusions between the polished
surfaces of the GJS cast iron (18201) and the chunky graphite cast iron (22348), which do not affect the surface state. Images with a wrong surface
state have a clearly lower surface state confidence (mean 80–91 %) than correctly classified images (99.9–100 %).

The settings were chosen on the validation accuracy with DINOv3 ViT-B/16 features, five seeds per setting.
Starting from 5000 episodes and a learning rate of `1e-4` (otherwise the current defaults: 97.5 ± 0.2 %),
one setting was changed at a time. A setting was changed only if it raised the mean validation accuracy by
more than the larger of the two standard deviations:

- **Learning rate and number of episodes** matter most: with `3e-5`, 500 episodes reach only 94.6 % (10,000:
  97.2 %), with `1e-4` 96.6 %. A learning rate of `1e-3` with 2000 episodes was the most accurate (97.8 ± 0.2 %),
  also with five further seeds (97.7 ± 0.3 % vs. 97.3 ± 0.2 % for the starting point), and became the default.
  More episodes do not help at this learning rate (5000 and 10,000 episodes: 97.5 %). Training takes about
  1 minute on a laptop CPU.
- **Network size**: the embedding dimension hardly matters (2 to 32 dimensions: 97.3–97.7 %), 8 gave the most
  consistent results. 1, 2 and 4 hidden layers perform equally well (97.3–97.6 %), 2 the most consistently; a
  purely linear projection (0 hidden layers: 96.9 %) and 8 hidden layers (96.6 %) are worse.
- **Episode composition** (5, 9 or 15 support and 1, 5 or 10 query images per class: 97.2–97.5 %) and
  **weight decay** (0 to `1e-1`: 97.4–97.5 %) hardly matter. The cosine learning rate schedule makes the
  results more consistent between seeds (constant learning rate: 97.4 ± 0.4 %).
- **Feature extractor**: ViT-H+/16 also met the criterion (97.8 ± 0.3 %), but not in the surface state
  (99.80 ± 0.09 % vs. 99.91 ± 0.03 %), and it is about eight times slower (see above).
- With the final settings, the surface state accuracy on the test set rises from 99.51 ± 0.08 % (earlier
  settings: 500 episodes, `N_S = 9`, `N_Q = 1`) to 99.86 ± 0.01 %, and the number of test images with a wrong
  surface state drops from 60–85 to 19–23.

**Class grouping**, compared on the surface state accuracy with the final settings: one class per material and
surface state reaches 99.91 ± 0.05 % (validation) and 99.86 ± 0.01 % (test). Pooling the materials into one
class per surface state reaches 99.87 ± 0.05 % and 99.85 ± 0.02 % (`N_S = 5`, `N_Q = 10`) or 99.93 ± 0.06 % and
99.87 ± 0.02 % (`N_S = 15`, `N_Q = 30`, as many images per episode as with six classes). As neither grouping is
more accurate by more than one standard deviation, one class per material and surface state stays the default.
All test images come from the three training materials, so the comparison does not show how either grouping
generalises to new materials.

## Installation

Python 3.12 or newer is required.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# OR Linux / macOS:
# source .venv/bin/activate

# Install PyTorch for your hardware, see https://pytorch.org/get-started/locally/
# e.g. CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# OR NVIDIA GPU with CUDA 13.0 (cu126 and cu132 are also available; check your driver with nvidia-smi).
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

### DINOv3 weights (only needed for feature extraction)

The DINOv3 weights are distributed by Meta under the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) and are not part of this
repository. Request access on the [DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
and place the weight files in `src/Models/`:

| `FeatureExtractor` | Model     | Weight file                                          | Feature dimension |
|--------------------|-----------|------------------------------------------------------|-------------------|
| `DINOv3`           | ViT-B/16  | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`       | 768               |
| `DINOv3-H16plus`   | ViT-H+/16 | `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`   | 1280              |

The DINOv3 model code is loaded through `torch.hub`, pinned to a fixed commit of the DINOv3 repository
(internet access is needed on first use).

## Data

The microscope images are available at **[link to the image data set]**.

All data sets contain three materials. A material is named `<BAM sample number>-<Material>`; the name must not
contain an underscore, as it is part of the class names (`<Material>_<SurfaceState>`) and the feature file names.

| Material                    | Description                     |
|-----------------------------|---------------------------------|
| `18201-GJSCastIron`         | GJS cast iron                   |
| `22116-90MnCrV8Unhardened`  | 90MnCrV8 tool steel, unhardened |
| `22348-ChunkyGraphite`      | Chunky graphite cast iron       |

Feature files are included in the repository, so the network can be trained without the images:

| Data set          | Images per material and class (training / testing)              | Included features    |
|-------------------|-----------------------------------------------------------------|----------------------|
| `CompleteDataSet` | 1500 / 2000 (polished), 1500 / 3000 (scratched)                 | `DINOv3`, `DINOv1`   |
| `SmallDataSet`    | 100 / 100                                                       | `DINOv1`             |
| `DummyDataSet`    | 10 / 10                                                         | `DINOv1`             |

The `DINOv1` files contain the DINOv1 ViT-B/16 features used in the thesis. They can be used for training
and comparisons, but can't be regenerated with this code.

To extract features yourself, place the tiles (224 × 224 pixels, cut from the microscope images at full
resolution) in the following layout:

```
src/DataSets/<DataSet>/Images/<Material>/<SurfaceState>/<Training|Testing>/*.tif
                              e.g. 18201-GJSCastIron/Polished/Training/
```

## Usage

All settings are defined at the top of each script, in the `ControlVariables` data class.

**1. Feature extraction** (optional, needs the images and the DINOv3 weights):
set `DataSet` and `FeatureExtractor` in `src/PreprocessImages.py` and run

```bash
python src/PreprocessImages.py
```

This writes `src/DataSets/<DataSet>/<DataSet>_<Material>_<FeatureExtractor>Features.h5`, one file per
material. On a laptop CPU, ViT-B/16 processes about 4–5 images per second, ViT-H+/16 about 0.5.

**2. Training and testing**: set `DataSet`, `FeatureExtractor` and the training parameters in
`src/TrainNetwork.py` and run

```bash
python src/TrainNetwork.py [--run-name <RunName>]
```

Every run writes its results to `src/Outputs/<RunName>/` (by default the start date and time; an existing folder is
never overwritten). The other scripts load the network with `--network <RunName>`:

| File                                       | Content                                                              |
|--------------------------------------------|----------------------------------------------------------------------|
| `Settings.json`                            | All settings of the run, including the seed (drawn at random with `GenerateRandomSeed = True`) |
| `Training.log`                             | Network structure, validation split, training statistics and test results. With `WriteDetailedDebugInfo = True` also every episode, the loss and accuracies per episode and the surface state confidence of every test image |
| `TrainingProcess.csv`, `TrainingInfo.png`  | Loss, training and validation accuracy per episode                   |
| `ConfusionMatrix.csv`                      | Test confusion matrix (rows: true class, columns: predicted class)   |
| `PCA_DinoFeatures.png`, `PCA_Embeddings.png` | PCA of the test features and of their embeddings with the prototypes. Misclassified test images are outlined: black for a wrong surface state, grey for a wrong material only |
| `PCA_Interactive.html`                     | Both PCAs to explore in a web browser (no internet needed): zoom, filters, and the tile image, true and predicted class on hover. Shows all misclassified and `InteractivePCAImages` (1000) randomly chosen correctly classified test images. Switched off with `InteractivePCA = False` |
| `SurfaceStateScores.png`                   | Surface state score of every test tile per material: distance to the nearest polished minus to the nearest scratched prototype. A tile is scratched from 0 on, so the plot shows how clearly the states are separated |
| `TrainedProtoNet.pth`, `PrototypeEmbedding.pth`, `Labelmap.json` | Trained encoder weights, class prototypes and the label of each class |

Runs with the same settings and seed give identical results on the same machine and software versions.

**3. Classifying complete microscope images** (needs the DINOv3 weights and a trained network):
`src/ClassifyImage.py` cuts every image into 224 × 224 tiles (99 tiles for a 2464 × 2056 pixel RASI image,
centred grid), classifies the surface state of every tile, and rates the image as scratched if more than 20 %
of its tiles are scratched (provisional; in the thesis, polished images had at most 5 % scratched tiles).

```bash
python src/ClassifyImage.py <image or folder> [...] --network <RunName> [--threshold 20] [--json] [--no-tilemap]
```

`--network` is the output folder of a training run in `src/Outputs/`; the feature extractor is taken from its
`Settings.json`. The results are written to `src/Outputs/Classification_<date and time>/` (or `--output`):
`Classification.csv` (one row per image), `Tiles.csv` (one row per tile), `Classification.log`, and a tile map
per image with the scratched tiles marked orange and the polished tiles blue. `--json` also prints the results
as JSON, e.g. for a control program.

In a loop, e.g. on the RASI platform, use the classifier from Python so that the models are loaded only once:

```python
from ClassifyImage import ImageClassifier

Classifier = ImageClassifier("<RunName>")           # loads the network and DINOv3 once
Result     = Classifier.Classify("Image.tif")       # about 20 s per image with ViT-B/16 on a laptop CPU
if Result["Result"] == "Scratched":
    ...
```

| Result field                  | Meaning                                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------|
| `Result`                      | `Polished` or `Scratched`                                                               |
| `ScratchedPercent`            | Share of scratched tiles, compared with `ThresholdPercent`                              |
| `ScratchedTiles`, `Tiles`     | Number of scratched tiles and of all tiles                                              |
| `MeanStateConfidencePercent`  | Mean surface state confidence of the tiles                                              |
| `LowConfidenceTiles`          | Tiles with a surface state confidence below 90 %. Many of them point to an unusual image (e.g. a new material or a focus problem) |
| `TilesToChangeResult`         | Number of tiles that would have to change their state to change the result: a measure of how clear the decision is |
| `TileStates`, `TileConfidencesPercent`, `TileBoxes` | State, confidence and position of every tile                      |

The trained network files (`.pth`) are not tracked by git; copy the run folder to the computer that classifies
the images.

**4. Adding prototypes of a new material** (optional, needs the DINOv3 weights and a trained network):
the trained prototypes often classify a new material well already. For a material with a clearly different
microstructure, `src/AddNewMaterialPrototype.py` adds new prototypes without retraining the network (few-shot, the
encoder is not changed). Put the RASI images of every reference surface in its own folder, without further levels:

```
src/DataSets/AdditionalPrototypes/<Material>_<Reference>_<State>/*.tif     e.g. 22188-GJS400CastIron_1200Grit_Scratched
src/DataSets/AdditionalPrototypes/<Material>_<State>/*.tif                 e.g. 22188-GJS400CastIron_Polished
```

```bash
python src/AddNewMaterialPrototype.py <Material> --network <RunName> [--skip-existing-check]
```

Every folder becomes one prototype, the mean embedding of the tiles of its images, so separate references (e.g. one per
grit) show which scratches remain. The script first checks whether the existing prototypes already classify at least
95 % of the reference tiles correctly; then nothing is added (`--skip-existing-check` continues anyway). Otherwise the
network with the existing and the new prototypes is written to `src/Outputs/<RunName>+<Material>/` and is used like any
training run, e.g. `ClassifyImage.py --network <RunName>+<Material>`. The log `AddNewMaterialPrototype.log` lists
the check. `PCA_NewPrototypes.png` and the interactive `PCA_NewPrototypes.html` (with a preview of
every reference tile) show the reference tiles and the new prototypes in the PCA of the test set (the same PCA as in the
training run), also when nothing is added. For these, the test set of the training materials is classified with the
existing and the new prototypes (`TestSet.log`, `ConfusionMatrix.csv`); the log reports how many test images are closest
to a new prototype. The prototypes depend on the encoder: after retraining, add them again from the same images.

**5. Generating a polishing recipe** (`src/RecipeGenerationLoop.py`): the operator enters the material removal rates of
the 9, 3 and 1 µm stages [µm/s] and the target removal depth per slice. Every trial is the full sequence on the RASI
platform: 9 µm for a fixed time (20 s; removes the earlier scratches and leaves defined 9 µm scratches, so every trial
starts alike), then 3 µm and 1 µm for the trial's times, then images at several positions. `RecipeGenerator.Check`
classifies them: polished means at least 90 % of all tiles polished, every image polished and at most 20 % uncertain
tiles. The start times (60 s for both stages) are doubled until they polish. Then the 3 µm and 1 µm times are bisected
alternately: each stage keeps its longest failed and shortest polished time, and the next trial tries the middle with
the other stage at its shortest polished time, until both intervals are at most 10 s (at most 12 trials). The recipe is:

- 3 µm and 1 µm: the shortest polished times + 15 s each, so deviations between slices are still polished
- 9 µm: (target depth − removal of the 3 and 1 µm stages) / 9 µm removal rate

```python
from RecipeGenerationLoop import RecipeGenerator, ControlVariables

Controls  = ControlVariables(RemovalRate9um_um_s=0.05, RemovalRate3um_um_s=0.01, RemovalRate1um_um_s=0.003, TargetDepth_um=10)
Generator = RecipeGenerator("<RunName>", "22188_Sample1", Controls)    # loads the network and DINOv3 once
Answer    = {"Decision": "Trial", "NextTrial": Generator.FirstTrial()}
while Answer["Decision"] == "Trial":
    Polish(Answer["NextTrial"]); Images = AcquireImages()              # RASI: {"9um": s, "3um": s, "1um": s}, then images
    Answer = Generator.Check(Images)                                   # "Trial", "Done" (Recipe.json) or "NotPolished"
```

The decision is made automatically, without input from the operator. `NotPolished` means the surface was not polished
even at the time limit (300 s of 3 µm or 1 µm polishing). All of these values are set in its `ControlVariables`. The results are written to `src/Outputs/Recipe_<SampleName>_<date and time>/`:
`Trials.csv`, `RecipeGeneration.log`, a tile map of every image, and `Recipe.json` with the polishing times and the
removal per stage. Compared with the thesis, the search is a proper bisection (the thesis added or subtracted half of the
current time, which does not converge on the shortest time) and polishing times are not limited to 30 s blocks.

`src/RASIExample.py` shows how the RASI control script uses it, moving the images of every trial away so the next
trial only sees its own. The machine calls are collected in one class, so the example runs as a simulation with a
given "true" shortest time per stage and folders of polished and scratched images:
`python src/RASIExample.py --network <RunName> --rates <R9> <R3> <R1> --depth <µm> --polished <folder> --scratched <folder> [--minimum <T3> <T1>]`.
On the machine, fill in `RASI.Polish` and `RASI.AcquireImages` with the calls of the control software.

**6. Comparison with the conventional method** (`src/HoughComparison.py`): the Hough transform method of the thesis,
applied to whole micrographs and independent of the network. Edges are detected (Canny) and every edge pixel votes for
the straight lines through it; each peak of the Hough accumulator is a line, with the number of edge pixels on it as its
strength. The Hough value of an image is the mean strength of its 10 strongest lines, and the image is scratched from a
Hough value of 300 on, for all materials. Compared with the thesis, it runs in Python only (no ImageJ) and detects the
edges on the grey image instead of a binarised one (which also removed faint scratches). Two further changes:

- The contrast is stretched over the matrix only. Dark graphite would otherwise use up the contrast range, and the faint
  scratches next to it would give no edges.
- The 10 strongest lines count instead of only the strongest one. A polished micrograph often has one or two isolated
  scratches, which give one strong line; a scratched surface gives many.

```bash
python src/HoughComparison.py <image or folder> [...] [--threshold 300]
```

Images whose file name contains `polished` or `scratched` are scored against that label, per material (the file name up
to the first `_`). Results in `src/Outputs/Hough_<date and time>/`: `HoughComparison.log`, `Hough.csv`, the Hough values
of all images (`HoughValues.png`) and a Hough diagram of every image (image, edges with the 10 strongest lines shaded from
dark for the strongest to light, and the Hough space). On 60 whole micrographs of the three training materials (10 per
material and state), it classifies all 60 correctly (`src/Outputs/Hough_2026-10-19_16-13-32`).

**The Hough method is tuned for these materials.** The matrix stretch, the number of lines and the threshold were chosen
on the same 60 micrographs; they are not validated on other micrographs. Without the matrix stretch, 4 of the 10
scratched micrographs of the chunky graphite cast iron (22348) were rated polished. Every new material, especially one
with another microstructure, would need this tuning again, whereas the network needs no handcrafted image processing.

## Repository structure

```
src/
├── PreprocessImages.py       Feature extraction with DINOv3
├── TrainNetwork.py           Training, validation and testing of the prototypical network
├── ClassifyImage.py          Classification of complete microscope images (e.g. in the RASI loop)
├── AddNewMaterialPrototype.py  Prototypes of a new material from a few reference images (optional)
├── RecipeGenerationLoop.py   Polishing recipe: shortest 3 and 1 µm times and the 9 µm time for the depth (called by RASI)
├── RASIExample.py            Example of the RASI control loop around RecipeGenerationLoop.py (with a simulation)
├── HoughComparison.py        Conventional comparison method of the thesis (Hough transform on whole micrographs)
├── DataSets/                 Feature files (and images, not tracked)
├── Models/                   DINOv3 weights (not tracked)
└── Modules/
    ├── DataSetHandling.py    Loading images and feature files
    ├── EpisodeHandler.py     Validation split and episode sampling
    ├── NeuralNetwork.py      DINOv3 wrapper, embedding network and loss
    ├── NetworkTesting.py     Test set evaluation and PCA
    ├── InteractivePCA.py     Interactive PCA of the test set (HTML page)
    └── Logging.py            Log files, tables and plots
```

## Citation

This code is based on the master's thesis

> J. Hornig, *Automated Surface Quality Optimization for Robot-Assisted Automated Serial-Sectioning and
> Imaging*, Master's thesis, [University], Berlin, 2026.

## License

The code is released under the [MIT License](LICENSE). The DINOv3 model and weights are subject to the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
