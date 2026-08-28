# Loblolly Pine Tracheid Detection Pipeline 
A pipeline for the detection and classification of cells in transverse section of Loblolly pine (Pinus taeda). Developed to examine changes in tracheid anatomy across changes in elevation (marsh vs. reference zones), time (Recent vs. Old growth), and salinity
# Pipeline overview
stitch → tissue mask → deconvolve → segment → features → adjacency → classify → visualize
# Setup
Install [uv](https://docs.astral.sh/uv/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
# Clone the repository
```bash
git clone https://github.com/Sehrin1905/loblolly-pipeline.git
cd loblolly-pipeline 
```
# Usage
``
## 1. Stitch tiles into a mosaic 
Insert SAMPLE_NAME(i.e. Reference_636_Old) 
```bash
uv run scripts/stitch.py data/SAMPLE_NAME/tiles/ --out data/SAMPLE_NAME/mosaic.tif
```
## 2. Run the tissue mask: The Black and White Map 
```bash
uv run scripts/tissue_mask.py data/SAMPLE_NAME/mosaic.tif data/SAMPLE_NAME/tissue_mask.png
```
## 3. Color deconvolution: Separate two stains from color into grayscale maps 
```bash
uv run scripts/deconvolve.py data/SAMPLE_NAME/mosaic.tif
```
## 4. Segmentation of cell lumens: Find and label every individual cell lumen 
```bash
uv run scripts/segment.py data/SAMPLE_NAME/mosaic.tif data/SAMPLE_NAME/tissue_mask.png data/SAMPLE_NAME/labels.npy
```
## 5. Features: Measures size, shape, eccentricity, and wall thickness for every cell lumen 
```bash
uv run scripts/segment.py data/SAMPLE_NAME/mosaic.tif data/SAMPLE_NAME/tissue_mask.png data/SAMPLE_NAME/labels.npy
```
## 6. Cell Adjacency 
```bash
uv run scripts/features.py data/SAMPLE_NAME/labels.npy data/SAMPLE_NAME/features.csv
```
## 7. Classification: Ray, Resin duct, or Tracheid
```bash
uv run scripts/classify.py data/SAMPLE_NAME/features.csv data/SAMPLE_NAME/classified.csv
```
## 8. Visualization 
```bash
uv run scripts/visualize.py data/SAMPLE_NAME/mosaic.tif data/SAMPLE_NAME/labels.npy data/SAMPLE_NAME/classified.csv data/SAMPLE_NAME/overlay.png
```
# Notes 
## Sample Naming Convention 
Samples are named '[Site]_[ID]_[Time]
- Site: 'Marsh' or 'Reference' 
- ID: Tree identifier number (i.e. Reference 639 or Marsh 112)
- Time: 'Recent' (youngest growth rings) or 'Old' (oldest growth rings)
## Dependencies 
- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation 
- OpenCV - stitching 
- scikit-image - morphology and shape distincion 
- pandas - feature tables 
- matplotlib - figures 
Dependencies are managed with 'uv' and tracked in 'pyproject.toml' 
## Additional notes 
- For samples with 50+ tiles, resize tiles to 50% before stitching to avoid memory issues 
- For samples with 80+ tiles use Google Colab for stitching (or a bigger computer)
- If you resize the tiles AND the mosaic cellpose might not be able to detect all the cells you need it to (cell numbers will be an artifact of the pipeline not biology) 
- Run caffeinate during the segmentation step to prevent your Mac from sleeping!!! 
- The mosaic has to be color (RGB) not grayscale for deconvolution (because you're separating the stains into two grayscale maps so you can't start out with a grayscale map for deconvolution) 
- Similar to the naming convention, the output for each sample (i.e. the tissue mask, overlay, etc.) is stored in 'data/SAMPLE_NAME/'
## Potential Issues 
- Segmentation time increases proportionally with mosaic size (large mosaics can take at least 1 - 2.5 days)