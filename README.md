# <center> End-to-end learning for music audio exploration <center/>
<center> <big> Semester Project - LTS2 Laboratory - EPFL <big/> <center/>
<center> <small> Justine Weber <small/> <center/>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JustineWeb/pds-wavenet.git/master)

The goal of this project is to get a better idea on the way artificial neural networkscan be trained using raw waveforms as input, for machine learning tasks related tomusic. The original purpose of this study was to get an insight on WaveNet, one such DeepLearning algorithm for music generation.  Through the project, we have decided to take another path and focus on another algorithm intended for music classification, based on the following paper : _End-to-end learning for music audio tagging at scale_. This project has mainly been focused on implementing the solution suggested in the latter paper and trying to train the model. 

In this repository, you can find all code that is used in the last version of the project. 
The repository is organised in the following way :
- `PDS_final.ipynb` contains all code for experiments in a jupyter notebook, with some graphs and results.
- folder `python_files` contains all functions that are used in the jupyter notebook, organized in several files depending on their usefulness.
- `requirements.txt` is the list of libraries you should install in order to run the code.


In order to understand the purpose of this work, I recommend reading the corresponding report : report.pdf.

Installation setup in the environment (python 3.6)
pip install librosa
pip install tensorflow
pip install pandas
pip install matplotlib
pip install tqdm


%
