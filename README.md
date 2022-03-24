# CycleSense

CycleSense model that is utilized in the [SimRa Project](https://simra-project.github.io) for automatic detection of (near-miss) incident.
<img src=https://smart-city-berlin.de/fileadmin/_processed_/2/6/csm_Simra_318b76c192.jpg width="500" />

This repository includes the preprocessing pipeline, the CycleSense model, as well as the other models that were used for evaluation.
All of these models can be found in the [classifier folder](https://github.com/tritter1612/simra-incident-detect/tree/master/classifier).
The modified [heuristic](https://www.sciencedirect.com/science/article/pii/S157411922030064X) and a reimplementation of the FCN proposed by [Sánchez](https://upcommons.upc.edu/handle/2117/191781) is also available there.

This model is an adaptation of the [DeepSense model](https://github.com/yscacaca/DeepSense) introduced by [Yao et al. (2017)](https://dl.acm.org/doi/abs/10.1145/3038912.3052577).
It was officially introduced in the paper "Detecing (Near-miss) Incidents in Bicycle Traffic from Mobile Motion Sensors" in the [Journal of Pervasive and Mobile Computing](https://www.journals.elsevier.com/pervasive-and-mobile-computing).

## Getting Started

If you want to make use of the CycleSense model you need to install the dependencies first.
You can do that using Anaconda or Pip with one of the following commands:

### Anaconda

```bash
conda env create -f conda-cpu.yml
```

### Pip
```bash
pip install -r requirements.txt
```

## Training the model

If you want to retrain the model, you need to reformat the ride files obtained from the [SimRa dataset](https://github.com/simra-project/dataset) first.

```bash
python export.py <source_dir> <target_dir>
```

Thereby, the ```target_region``` parameter is optional and per default set to Berlin, as this is the region containing the most rides.
If more than one region should be formatted, this command can be executed multiple times.
The ```lin_acc_flag``` is False as a default, as including the linear accelerometer data did not lead to any improvements but required more memory consumption and training time in our experiments.

In a next step, the data need to be preprocessed.
The defined preprocessing pipeline can be executed via the following command:

```bash
python preprocess.py <dir>
```

There are many configuration options available that can be found using the help functions 
```python preprocess.py -h```.
It is noteworthy, that the utilization of the Generative Adversarial Network for data augmentation can be deactivated, as well as the fourier transform.

Finally, the model can be retrained with the command:

```bash
python cyclesense.py <dir>
```

## Evaluation

The CycleSense model does clearly outperform all the other models we have encountered for evaluation on the more recent Android rides of the [SimRa dataset](https://github.com/simra-project/dataset), as illustrated in the figure below.

<img src=media/roc_auc_results.png width="500" />