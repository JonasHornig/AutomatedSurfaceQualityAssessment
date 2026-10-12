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
   (768 dimensions for ViT-B/16) through `NumberOfHiddenLayers` hidden blocks (Linear, BatchNorm, LeakyReLU)
   and a linear output layer into a low dimensional embedding space (8 dimensions by default).
3. **Episodic training** (`src/TrainNetwork.py`): every episode draws `N_C` classes with `N_S` support and
   `N_Q` query images each. The prototype of a class is the mean embedding of its support images, and the
   loss is the cross entropy over the negative squared Euclidean distances of the queries to the prototypes.
4. **Validation**: a fixed, seeded fraction of the training images of every class (`ValidationFraction`)
   is held back and never used as support or query image. It is used to monitor the training.
5. **Testing**: after training, the final prototypes are computed from all training images (without the
   validation images), and each test image is assigned to the class of the nearest prototype.

A class is the combination of material and surface state, e.g. `18201_Polished`.

## Results

`CompleteDataSet`, default settings of `TrainNetwork.py` (500 episodes, `N_C = 6`, `N_S = 9`, `N_Q = 1`,
8 embedding dimensions), mean ± standard deviation over five seeds, 15,000 test images:

| Features        | Validation accuracy | Test accuracy, all classes | Surface state (polished vs. scratched) |
|-----------------|---------------------|----------------------------|----------------------------------------|
| DINOv3 ViT-B/16 | 94.7 ± 0.5 %        | 95.8 ± 0.2 %               | 99.4 %                                 |
| DINOv1 ViT-B/16 | 96.2 ± 0.3 %        | 96.9 ± 0.2 %               | 99.8 %                                 |

Most remaining errors are confusions between the polished surfaces of the materials 18201 and 22348.
The training settings were originally tuned on the DINOv1 features.

The embedding dimension was chosen on the validation accuracy (DINOv3 ViT-B/16, five seeds each):
2 dimensions are too few (92.8 ± 2.4 %), 3 to 8 dimensions perform equally well (94.5–94.8 %),
and 16 or 32 dimensions are slightly worse (94.4 and 94.0 %). 8 dimensions gave the most consistent results.

## Installation

Python 3.12 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux / macOS: source .venv/bin/activate

# Install PyTorch for your hardware first, see https://pytorch.org/get-started/locally/
# e.g. CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

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
                              e.g. 18201/Polished/Training/
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
python src/TrainNetwork.py
```

Every run writes its results to `src/Outputs/<RunName>/` (by default the start date and time):

| File                                       | Content                                                              |
|--------------------------------------------|----------------------------------------------------------------------|
| `Settings.json`                            | All settings of the run, including the seed                          |
| `Training.log`                             | Network structure, validation split, training statistics, test results |
| `TrainingProcess.csv`, `TrainingInfo.png`  | Loss, training and validation accuracy per episode                   |
| `ConfusionMatrix.csv`                      | Test confusion matrix (rows: true class, columns: predicted class)   |
| `PCA_DinoFeatures.png`, `PCA_Embeddings.png` | PCA of the test features and of their embeddings with the prototypes |
| `TrainedProtoNet.pth`, `PrototypeEmbedding.pth`, `Labelmap.json` | Trained encoder weights, class prototypes and the label of each class |

Runs with the same settings and seed give identical results on the same machine and software versions.

## Repository structure

```
src/
├── PreprocessImages.py       Feature extraction with DINOv3
├── TrainNetwork.py           Training, validation and testing of the prototypical network
├── DataSets/                 Feature files (and images, not tracked)
├── Models/                   DINOv3 weights (not tracked)
└── Modules/
    ├── DataSetHandling.py    Loading images and feature files
    ├── EpisodeHandler.py     Validation split and episode sampling
    ├── NeuralNetwork.py      DINOv3 wrapper, embedding network and loss
    ├── NetworkTesting.py     Test set evaluation and PCA
    └── Logging.py            Log files, tables and plots
```

## Citation

This code is based on the master's thesis

> J. Hornig, *Automated Surface Quality Optimization for Robot-Assisted Automated Serial-Sectioning and
> Imaging*, Master's thesis, [University], Berlin, 2026.

## License

The code is released under the [MIT License](LICENSE). The DINOv3 model and weights are subject to the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
