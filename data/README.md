ChiSCO Dataset
This project uses the ChiSCO (Chinese Imagined Speech Corpus) dataset.
Dataset Attribution
Created by:

Zihan Zhang¹, Xiao Ding¹, Yu Bao¹, Yi Zhao¹, Xia Liang², Bing Qin¹, Ting Liu¹

¹ Harbin Institute of Technology, China
² Northeast Forestry University, China
Publication:

Zhang, Z., Ding, X., Bao, Y., Zhao, Y., Liang, X., Qin, B., & Liu, T. (2024).
Chisco: An EEG-based BCI dataset for decoding of imagined speech.
Scientific Data, 11(1), 1265.
DOI: 10.1038/s41597-024-04114-1

Dataset Repository: OpenNeuro ds005170

Download Instructions

Visit OpenNeuro dataset ds005170
Download preprocessed .pkl files for subjects 1-5
Place in this structure:

   data/
   ├── sub-01/
   ├── sub-02/
   ├── sub-03/
   ├── sub-04/
   └── sub-05/

Dataset Details
Recording Specifications

Subjects: 3 (S1: F 26y, S2: M 30y, S3: M 22y)
Trials: 6,681 per subject
Duration: 900+ min per subject
Channels: 125 EEG + 6 external (EOG, mastoids)
Sampling: 1000 Hz → downsampled to 500 Hz

Experimental Paradigm

Read sentence: 5s
Imagine speaking: 3.3s
Rest: 1.8s
Total: 10.1s per trial

Text Materials

Language: Chinese (6-15 characters)
Sentences: 6,681 everyday expressions
Categories: 39 semantic types


Inter-Subject Correlation
From ChiSCO paper:

r(S1, S2) = 0.126
r(S2, S3) = 0.147
r(S1, S3) = 0.168

Interpretation: ~87% subject-specific variance → cross-subject learning is hard!

Citation
bibtex@article{zhang2024chisco,
  title={Chisco: An EEG-based BCI dataset for decoding of imagined speech},
  author={Zhang, Zihan and Ding, Xiao and Bao, Yu and Zhao, Yi and Liang, Xia and Qin, Bing and Liu, Ting},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={1265},
  year={2024},
  doi={10.1038/s41597-024-04114-1}
}

License: CC0 (Public Domain)
Note: Dataset files not included due to size. Download from OpenNeuro.

