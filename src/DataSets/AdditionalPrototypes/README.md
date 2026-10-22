# Additional prototypes

Reference images of a new material, to add its prototypes to a trained network without retraining it.
This is optional: it is mainly worth it for a material whose microstructure is clearly different from the
training materials.

## Folder structure

One folder per reference directly in this folder, without further folder levels:

```
AdditionalPrototypes/
├── <Material>_<Reference>_<State>/      e.g. 22188-GJS400CastIron_1200Grit_Scratched/
│   ├── Image1.tif
│   └── ...
└── <Material>_<State>/                  e.g. 22188-GJS400CastIron_Polished/
    └── Image1.tif
```

- `<Material>` and `<Reference>` must not contain `_`. Every folder of a material starts with the same `<Material>`.
- `<State>` is a surface state of the network: `Polished` or `Scratched`.
- `<Reference>` is optional and names the scratch type, e.g. `600Grit`, `1200Grit`, `9um`, `3um`.
- Images: RASI at 50x (0.107 µm/px), `.tif`, `.tiff`, `.png`, `.jpg` or `.jpeg`, in focus, with representative
  microstructure and, for `Scratched`, scratches over the whole image. One image per folder can be enough.
- Every folder becomes one prototype: the mean embedding of all tiles of its images.

## Adding the prototypes

From the repository root:

```
python src/AddNewMaterialPrototype.py <Material> --network <training run in src/Outputs/>
```

e.g. `python src/AddNewMaterialPrototype.py 22188-GJS400CastIron --network RASI_ViTB`

- Step 1: if the existing prototypes already classify at least 95 % of the reference tiles correctly, nothing is
  added. `--skip-existing-check` continues anyway.
- Step 2: otherwise every image is held out once for validation. If the new prototypes reach 95 %, the network
  with them is written to `src/Outputs/<training run>+<Material>/`. `--ignore-validation` adds them even below
  95 %. Use the new network like a training run, e.g.
  `python src/ClassifyImage.py <image> --network <training run>+<Material>`.
- The results are in `AddNewMaterialPrototype.log` in that folder, and `PCA_NewPrototypes.png` and the interactive
  `PCA_NewPrototypes.html` show where the reference tiles and the new prototypes lie compared with the training
  materials (written in every run).
- After retraining the network, run the script again with the same images.

The contents of this folder are not tracked by git, except this note.
