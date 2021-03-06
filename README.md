# AD Detection

Detecting Alzheimer's disease(AD) from spontaneous speech.

## Paper

Please cite:

- [1]	Liu Z, Guo Z, Ling Z, et al. Dementia Detection by Analyzing Spontaneous Mandarin Speech[C]//2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2019: 289-296.

- [2]	Liu Z, Guo Z, Ling Z, et al. Detecting Alzheimer's Disease from Speech Using Neural Networks with Bottleneck Features and Data Augmentation [C]// ICASSP 2021.


## Data Set

- [DementiaBank Pitt corpus](https://dementia.talkbank.org/access/English/Pitt.html).
- Unpublished mandarin dataset.


## Repository Structure
```
├── ad_detection
│   ├── dlcode                         # Deep learning model
│   ├── feature_cn                     # feature extraction of mandarin dataset
│   ├── feature_en                     # feature extraction of dementia bank dataset
│   ├── mlcode                         # Traditional meachine learing model
│   ├── prepare_cn                     # Data preprocessing for mandarin dataset
│   ├── prepare_en                     # Data preprocessing for dementia bank dataset
│   │   └── pylangacq_modified
│   └── tool                           # Tools
│       └── opensmile_scripts
├── notebook                           # Jupyter notebooks of Dementia bank dataset
│   └── log
├── notebook_cn                        # Jupyter notebooks of mandarin dataset
├── ws_cn                              # Workspace of mandarin dataset. Similiar as ws_en
└── ws_en                              # Workspace of Dementia bank dataset
    ├── data                           # Preprocessed data or extracted feature
    │   ├── bottleneck
    │   │   ├── high_pass
    │   │   └── hp_aug
    │   ├── speech_extract_b
    │   ├── speech_keep_b
    │   ├── tsv
    │   ├── tsv_sr
    │   ├── wav_mono
    │   ├── high-pass
    │   └── wav_hp_aug
    ├── data_ori                       # Original data from Pitt corpus
    │   ├── ctrl
    │   │   └── cookie
    │   └── dementia
    │       └── cookie
    ├── fusion                         # Merged features
    ├── label                          # summary.csv  blacklist.csv
    ├── list                           # Results of feature merging, K-fold division and prediction
    │   ├── result
    │   ├── result_aug
    │   ├── result_bottleneck
    │   ├── result_bottleneck_aug
    │       ......
    │   ├── split
    │   └── split_aug
    ├── mfa                            # Forced align using Montreal-Forced-Aligner(MFA)
    └── opensmile                      # Files generated by OpenSMILE toolkit
        ├── audio_features_CPE16_
            ......
        ├── audio_features_mfcc_
        └── audio_features_mfcc_aug

```


## Requirements and Installation
In order to run this code, you'll need the following packages.

* [openSMILE](https://www.audeering.com/opensmile/)
* [SoX](http://sox.sourceforge.net/)

And you need following python libraries.

* [TensorFlow](https://www.tensorflow.org/install/) >= 2.0 (include Keras)
* [NumPy](https://docs.scipy.org/doc/numpy/user/install.html)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [librosa](https://librosa.org/doc/latest/index.html)
* [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
* [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html)



## How to run

1. Download and prepare data following steps in `run_en.sh` (DementiaBank Pitt corpus)

2. Launch Jupyter and, using your browser, navigate to the URL shown in the terminal output (usually http://localhost:8888/)

3. 
    - Split data to 10 folds using `notebook/ML7-DataPrepareAndSplit.ipynb`
    - Train baseline models using `notebook/ML7-Test_acoustic_feature.ipynb`
    - Train deeplearing models with OpenSMILE LLDs: `notebook/DL2-OpenSMILE_LLDs.ipynb`
    - Train deeplearing models with bottleneck features: `notebook/DL2-LLDs-BottleNeck.ipynb`


## Note

The ASR model used for bottleneck feature extraction will not be made public.

The `*.py` files in `notebook` directory are auto generated from `*.ipynb` files.
You can find more information on 
[this page](https://github.com/ipython/ipython/issues/8009) and 
[this page](https://jupyter-notebook.readthedocs.io/en/stable/extending/savehooks.html).


## License

The MIT License; please see [LICENSE.txt](LICENSE.txt)
