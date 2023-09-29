# SpInOpt-MMD

This folder contains all the code to reproduce the experiments from our submission.

### Copyright

- The code in the ```jitkrittum``` folder is the original source code from [Jitkrittum et al.](https://github.com/wittawatj/interpretable-test).
- The code in the ```src``` folder is closely modeled on Jitkrittum et al., but it has been refactored and extended for SpInOpt-MMD.
- The experiments in the ```experiments``` folder are closely modeled on the experimental setting of Jitkrittum et al. for best comparability.

Therefore, we include the original license and copyright statement in the ```jitkrittum``` folder (```LICENSE_interpretable-test.txt```).  
Please note that this copyright applies to all parts on the code which are taken from Jitkrittum et al.

### Data
All data used in this analysis was obtained from public repositories (that may require registration for their access). 
These include: 

Benchmark data sets: 
- [Karolinska faces data](https://kdef.se/)
- [NIPS Paper dataset](https://www.kaggle.com/datasets/benhamner/nips-papers) 

Biological data sets
- ADNI data: MR images obtained from [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/) - check [this publication](https://proceedings.mlr.press/v149/bruningk21a.html) for details regarding patient selection and data preprocessing 
- PBMC data: data were obtained from [10X Genomics](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k) - check [this publication](https://doi.org/10.1038/nbt.3192) for details regarding data preprocessing.
- DRIAMS data: data were obtained from [the Dryad repository](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q) - check [this publication](https://doi.org/10.1038/s41591-021-01619-9) for details regarding data preprocessing for the two selected examples
- TCGA LGG whole genome sequencing data (ATRX mutation status) were obtained from [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)
- TCGA-LGG MR images were obtained from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=5309188)
