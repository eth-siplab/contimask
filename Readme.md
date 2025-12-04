# contimask

## Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time (NeurIPS 2025, Official Code)

Max Moebus, Björn Braun, Christian Holz<br/>

<p align="center">
</p>

---

> Explaining black-box models for time series data is critical for the wide-scale adoption of deep learning techniques across domains such as healthcare. Recently, explainability methods for deep time series models have seen significant progress by adopting saliency methods that perturb masked segments of time series to uncover their importance towards the prediction of black-box models. Thus far, such methods have been largely restricted to regular time series. Irregular time series, however, sampled at irregular time intervals and potentially with missing values, are the dominant form of time series in various critical domains (e.g., hospital records). In this paper, we conduct the first evaluation of saliency methods for the interpretation of irregular time series models. We first translate techniques for regular time series into the continuous time realm of irregular time series and show under which circumstances such techniques are still applicable. However, existing perturbation techniques neglect the timing and structure of observed data, e.g., informative missingness when data is not missing at random. Thus, we propose Contimask, a simple framework to also apply non-differentiable perturbations, such as simulating that parts of the data had not been observed using NeuroEvolution. Doing so, we successfully detect how structural differences in the data can bias irregular time series models on a real-world sepsis prediction task where 90% of the data is missing.

<p align="center">
  <img src="Figures/chaos_2.jpg">
</p>

Environment
----------

The environment is rather small (4 packages besides torch). You can create it using ```conda env create -f environment.yml```. Afterwards, you have to install torch yourseld. All experiments were performed using torch==2.7.0+cu128, torchaudio==2.7.0+cu128, torchvision==0.22.0+cu128. Adapt this to your set-up as needed.

Running
----------
The commands to run the experiments on artificial data:
```
bash main_runner.sh
```

The ```results.ipynb``` notebook then allows to visualize the results and calculate all metrics.

Overall Comments
----------
This code repository still needs an overhaul and in its current state only contains the files for the experiments on artificial data. The experiments on the sepsis data will follow soon after NeurIPS. We apologize for the delay.

The experiments for the value- and temp-based Rare Time and Rare Feature settings can be run indivdually via `python rare_feature_temp.py`, `python rare_feature_value.py`, `python rare_time_temp.py`, and `python rare_time_value.py`.

Our masking functions and perturbations are implemented in `attribution.mask_conti.py` and `attribution.perturbation_conti.py`.

Credits
----------
This repository is based largely on the structure of the [Dynamask](https://github.com/JonathanCrabbe/Dynamask) and [ExtremalMask](https://github.com/josephenguehard/time_interpret) repositories.