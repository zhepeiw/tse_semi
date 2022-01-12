# Codebase for Semisupervised Target Speaker Extraction

The repository contains two modules: the speaker embedding network and the target speaker extraction (TSE) network (Exformer).

# Requirements
  - speechbrain==0.5.10
  - pytorch==1.10
  - torchaudio==0.10
  - soundfile==0.10.3
  - wandb==0.12.5

The `wandb` package is for logging experimental metrics and artifacts and is not required.

# TSE
The training and evaluation scripts of the TSE are under the `tse/` directory. The configuration yaml files are under `tse/hparams/`. To train the extraction network, run

```bash
    cd ./tse
    python wsj_train.py hparams/sepformer_ann.yaml --experiment_name <experiment_name>
```

In this example, the `hparams/sepformer_ann.yaml` is the hyperparameter file of the additive exformer architecture trained with both `wsj0-2mix-extr` and the `VoxCeleb1` dataset with the pre-trained supervised model as the starting point. The argument `experiment_name` is a custom optional string.

If `wandb` is not installed, make sure to set the `use_wandb` flag to False; the experiments would be logged through the speechbrain logger.


To evaluate a trained model with wsj0-2mix-extr, run
```bash
    python wsj_train.py hparams/sepformer_ann_eval.yaml --use_wandb False --test_only True --output_folder <output_folder>
```
where the argument `<output_folder>` is as specified in the yaml file.


# SPID
The training and evaluation scripts of the speaker embedder network is contained in `spid/`. To train the speaker embedding network, run

```bash
    cd ./spid
    python spid_trainer.py -e blstm_softmax_8k_bs46_4sec
```

This script trains the BLSTM-based embedder network using the generalized end-to-end (GE2E) loss with the combination of `librispeech`, `VoxCeleb1`, and `VoxCeleb2` datasets. Please modify the entries under `datasets/speech` in the `spid/experiments/experiments_spid.py` file according to your preparation and location of the datasets.
