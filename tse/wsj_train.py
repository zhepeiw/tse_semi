#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.
To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml
The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.
Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
import datetime
import wandb
from spid_modules.ECAPA_TDNN import ECAPA_TDNN
from spid_modules.embedder import Embedder as BLSTM_Embedder
import pdb


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, enrollment, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)
        enrollment, enrol_lens = enrollment
        enrollment, enrol_lens = enrollment.to(self.device), enrol_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            # TODO: temporarily set to disable training mode for embedder
            if isinstance(self.hparams.Embedder, ECAPA_TDNN):
                if self.step == 1:
                    self.hparams.Embedder.eval()
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        #  emb = self.hparams.Embedder(enrollment)
        emb = self.compute_embedding(enrollment, enrol_lens)
        est_mask = self.hparams.MaskNet(mix_w, emb)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets, emb

    def compute_objectives(self, predictions, targets, enr_emb, src_masks, stage):
        """Computes the sinr loss
            args:
                predictions: [bs, L, nsrc]
                targets: [bs, L, nsrc]
                enr_emb: [bs, D]
                src_masks: [bs, D]
        """
        output_dict = {}
        if stage == sb.Stage.TRAIN:
            loss_nm_fn_dict = self.hparams.train_loss
        elif stage == sb.Stage.VALID:
            loss_nm_fn_dict = self.hparams.valid_loss
        else:
            loss_nm_fn_dict = self.hparams.test_loss

        for loss_nm, loss_fn in loss_nm_fn_dict.items():
            if loss_nm in ['si-snr']:
                output_dict[loss_nm] = loss_fn(targets, predictions, src_masks)
            elif loss_nm in ['triplet']:
                #  s1_emb = self.hparams.Embedder(predictions[:, :, 0])
                #  s2_emb = self.hparams.Embedder(predictions[:, :, 1])
                s1_emb = self.compute_embedding(predictions[:, :, 0], torch.ones(predictions.shape[0]).to(predictions.device))
                s2_emb = self.compute_embedding(predictions[:, :, 1], torch.ones(predictions.shape[0]).to(predictions.device))
                #  # rescale predictions
                #  # [bs, 1, 1]
                #  prediction_scales = 0.9 / torch.amax(torch.abs(predictions),
                #                                 dim=(1, 2), keepdim=True)
                #  scaled_predictions = prediction_scales * predictions
                #  # [bs, D]
                #  s1_emb = self.hparams.Embedder(scaled_predictions[:, :, 0])
                #  # [bs , D]
                #  s2_emb = self.hparams.Embedder(scaled_predictions[:, :, 1])
                output_dict[loss_nm] = loss_fn(enr_emb, s1_emb, s2_emb)
        loss = 0.
        for loss_nm, loss_val in output_dict.items():
            loss += self.hparams.loss_lambdas.get(loss_nm, 1.) * loss_val
        return loss, output_dict

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        enrollment = batch.enr_sig
        src_masks = torch.cat([batch.s1_clean.data, batch.s2_clean.data], dim=-1)
        src_masks = src_masks.to(self.device)

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.auto_mix_prec:
            with autocast():
                predictions, targets, enr_emb = self.compute_forward(
                    mixture, enrollment, targets, sb.Stage.TRAIN
                )
                loss, loss_dict = self.compute_objectives(predictions, targets, enr_emb, src_masks, sb.Stage.TRAIN)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # update datapoints info
                self.datapoints_seen += mixture.data.shape[0]
            else:
                pdb.set_trace()
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets, enr_emb = self.compute_forward(
                mixture, enrollment, targets, sb.Stage.TRAIN
            )
            loss, loss_dict = self.compute_objectives(predictions, targets, enr_emb, src_masks, sb.Stage.TRAIN)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
                # update datapoints info
                self.datapoints_seen += mixture.data.shape[0]
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        # additional training logging
        #  if self.hparams.use_wandb:
        #  self.train_loss_buffer.append(loss.item())
        #  if self.step % self.hparams.train_log_frequency == 0 and self.step > 1:
        #      self.hparams.train_logger.log_stats(
        #          stats_meta={"datapoints_seen": self.datapoints_seen},
        #          train_stats={'buffer-si-snr': np.mean(self.train_loss_buffer)},
        #      )
        #      self.train_loss_buffer = []
        if self.hparams.use_wandb:
            if len(loss_dict) > 1:
                loss_dict['total_loss'] = loss
            for loss_nm, loss_val in loss_dict.items():
                if loss_nm not in self.train_loss_buffer:
                    self.train_loss_buffer[loss_nm] = []
                self.train_loss_buffer[loss_nm].append(loss_val.item())
            if self.step % self.hparams.train_log_frequency == 0 and self.step > 1:
                self.hparams.train_logger.log_stats(
                    stats_meta={"datapoints_seen": self.datapoints_seen},
                    #  train_stats={'buffer-si-snr': np.mean(self.train_loss_buffer)},
                    train_stats = {'buffer-{}'.format(loss_nm): np.mean(loss_list) for loss_nm, loss_list in self.train_loss_buffer.items()}
                )
                self.train_loss_buffer = {}

        # very hacky update: model only keeps tracks of sisdr at the stage end
        if 'si-snr' in loss_dict:
            loss = loss_dict['si-snr']

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        enrollment = batch.enr_sig
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets, enr_emb = self.compute_forward(mixture, enrollment, targets, stage)
            loss, _ = self.compute_objectives(predictions, targets, enr_emb, None, stage)

        # Manage logging with wandb
        if stage == sb.Stage.VALID and self.hparams.use_wandb and self.step <= self.hparams.log_audio_limit:
            valid_update_stats = self.get_log_audios(mixture.data, targets.data, predictions.data)
            self.valid_stats.update(valid_update_stats)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_fit_start(self):
        super().on_fit_start()
        self.datapoints_seen = 0
        #  self.train_loss_buffer = []
        self.train_loss_buffer = {}
        self.valid_stats = {}
        self.load_pretrain_checkpoint()

    def on_stage_start(self, stage, epoch=None):
        super().on_stage_start(stage, epoch)
        if stage == sb.Stage.TEST:
            # disable normalization updates for test
            if isinstance(self.hparams.Embedder, ECAPA_TDNN):
                self.hparams.mean_var_norm_emb.update_until_epoch = 0

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.valid_stats.update(stage_stats)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr, "datapoints_seen": self.datapoints_seen},
                train_stats=self.train_stats,
                valid_stats=self.valid_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
            self.valid_stats = {}
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""
        self.on_evaluate_start(min_key='si-snr')
        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources
        from pesq import pesq

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        #  all_sdrs = []
        #  all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        all_pesqs = []
        all_pesqs_i = []
        #  csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        csv_columns = ["snt_id", "si-snr", "si-snr_i", "pesq", "pesq_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.test_dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    enrollment = batch.enr_sig
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets, enr_emb = self.compute_forward(
                            mixture, enrollment, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr, _ = self.compute_objectives(predictions, targets, enr_emb, None, sb.Stage.TEST)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture.data] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline, _ = self.compute_objectives(
                        mixture_signal, targets, enr_emb, None, sb.Stage.TEST
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    #  # Compute SDR
                    #  sdr, _, _, _ = bss_eval_sources(
                    #      targets[0].t().cpu().numpy(),
                    #      predictions[0].t().detach().cpu().numpy(),
                    #  )
                    #
                    #  sdr_baseline, _, _, _ = bss_eval_sources(
                    #      targets[0].t().cpu().numpy(),
                    #      mixture_signal[0].t().detach().cpu().numpy(),
                    #  )
                    #
                    #  sdr_i = sdr.mean() - sdr_baseline.mean()
                    #  sdr, _, _, _ = bss_eval_sources(
                    #      targets[0, :, 0].unsqueeze(0).cpu().numpy(),
                    #      predictions[0, :, 0].unsqueeze(0).cpu().numpy(),
                    #      compute_permutation=False
                    #  )
                    #
                    #  sdr_baseline, _, _, _ = bss_eval_sources(
                    #      targets[0, :, 0].unsqueeze(0).cpu().numpy(),
                    #      mixture_signal[0, :, 0].unsqueeze(0).cpu().numpy(),
                    #      compute_permutation=False
                    #  )
                    #  sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Compute PESQ
                    pesq_est = pesq(
                        fs=self.hparams.sample_rate,
                        ref=targets[0, :, 0].cpu().numpy(),
                        deg=predictions[0, :, 0].cpu().numpy(),
                        mode='nb'
                    )

                    pesq_baseline = pesq(
                        fs=self.hparams.sample_rate,
                        ref=targets[0, :, 0].cpu().numpy(),
                        deg=mixture_signal[0, :, 0].cpu().numpy(),
                        mode='nb'
                    )
                    pesq_i = pesq_est - pesq_baseline

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        #  "sdr": sdr.mean(),
                        #  "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                        "pesq": pesq_est,
                        "pesq_i": pesq_i,
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    #  all_sdrs.append(sdr.mean())
                    #  all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())
                    all_pesqs.append(pesq_est)
                    all_pesqs_i.append(pesq_i)

                row = {
                    "snt_id": "avg",
                    #  "sdr": np.array(all_sdrs).mean(),
                    #  "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                    "pesq": np.array(all_pesqs).mean(),
                    "pesq_i": np.array(all_pesqs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        #  logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        #  logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))
        logger.info("Mean PESQ is {}".format(np.array(all_pesqs).mean()))
        logger.info("Mean PESQi is {}".format(np.array(all_pesqs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

    def get_log_audios(self, mixture, targets, predictions):
        '''
            inp tensors with shape [bs, L, nsrc]
        '''
        targets = targets.permute(0, 2, 1)[0]  # [nsrc, L]
        predictions = predictions.permute(0, 2, 1)[0]  # [nsrc, L]
        mixture = mixture[0]
        # rescale to avoid clipping
        max_amp = max(
            *[x.item() for x in torch.abs(predictions).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        predictions = mix_scaling * predictions
        valid_stats = {
            'mix_{}'.format(self.step): mixture.cpu().numpy(),
            'pred_sp1_{}'.format(self.step): predictions[0].cpu().numpy(),
            'pred_sp2_{}'.format(self.step): predictions[1].cpu().numpy(),
            'sp1_{}'.format(self.step): targets[0].cpu().numpy(),
            'sp2_{}'.format(self.step): targets[1].cpu().numpy(),
        }
        valid_stats = {key: wandb.Audio(array, sample_rate=self.hparams.sample_rate) for key, array in valid_stats.items()}
        return valid_stats

    def load_pretrain_checkpoint(self):
        if self.hparams.pretrain_checkpointer is not None:
            best_ckpt = self.hparams.pretrain_checkpointer.find_checkpoint(
                min_key='si-snr'
            )
            for module_key in ['encoder', 'decoder', 'masknet', 'embedder']:
                param_file = best_ckpt.paramfiles[module_key]
                sb.utils.checkpoints.torch_parameter_transfer(
                    self.hparams.modules[module_key], param_file, self.device)
                print('Recovering pretrained {} from {}'.format(module_key, param_file))

    def compute_embedding(self, wavs, wav_lens):
        """Compute speaker embeddings.
        Arguments
        ---------
        wavs : Torch.Tensor
            Tensor containing the speech waveform (batch, time).
            Make sure the sample rate is fs=16000 Hz.
        wav_lens: Torch.Tensor
            Tensor containing the relative length for each sentence
            in the length (e.g., [0.8 0.6 1.0])
        """
        if isinstance(self.hparams.Embedder, ECAPA_TDNN):
            # normalize input by amplitude
            scales = 0.9 / torch.amax(torch.abs(wavs), dim=-1, keepdim=True)
            wavs = wavs * scales
            feats = self.hparams.compute_features(wavs)
            feats = self.hparams.mean_var_norm(feats, wav_lens)
            embeddings = self.hparams.Embedder(feats, wav_lens)
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device),
                epoch=self.hparams.epoch_counter.current
            )
            embeddings = embeddings / (1e-8 + torch.norm(embeddings, p=2, dim=-1, keepdim=True))
            return embeddings.squeeze(1)
        elif isinstance(self.hparams.Embedder, BLSTM_Embedder):
            embeddings = self.hparams.Embedder(wavs)
            return embeddings


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # setting up experiment stamp
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d+%H-%M-%S')
    if run_opts['debug']:
        time_stamp = 'debug+' + time_stamp
    stamp_override = 'time_stamp: {}'.format(time_stamp)
    overrides = stamp_override + '\n' + overrides if len(overrides) > 0 else stamp_override
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # setting up train logger
    if hparams['use_wandb']:
        print('Replacing logger with wandb...')
        from utils import MyWandBLogger
        hparams['wandb_logger']['initializer'] = wandb.init
        hparams['train_logger'] = MyWandBLogger(**hparams['wandb_logger'])

    run_opts["auto_mix_prec"] = hparams["auto_mix_prec"]

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    #  # Check if wsj0_tr is set with dynamic mixing
    #  if hparams["dynamic_mixing"] and not os.path.exists(
    #      hparams["base_folder_dm"]
    #  ):
    #      print(
    #          "Please, specify a valid base_folder_dm folder when using dynamic mixing"
    #      )
    #      sys.exit(1)

    # Data preparation
    from data.prepare_data import prepare_dummy_csv  # noqa
    run_on_main(
        prepare_dummy_csv,
        kwargs={
            "savepath": hparams["save_folder"],
        },
    )
    #  if 'wsj' in hparams['data_folder']:
    #      from data.prepare_data import create_wsj_csv
    #      run_on_main(
    #          create_wsj_csv,
    #          kwargs={
    #              "datapath": hparams['data_folder'],
    #              "savepath": hparams['save_folder'],
    #          }
    #      )
    from data.prepare_data import create_wsj_tse_csv
    for part in ['valid', 'test']:
        run_on_main(
            create_wsj_tse_csv,
            kwargs={
                "datapath": hparams['data_folder'],
                "txtpath": hparams['{}_txtpath'.format(part)],
                "savepath": hparams['{}_data'.format(part)],
            }
        )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from data.data_mixing import dynamic_mixing_prep, static_data_prep
        train_data = dynamic_mixing_prep(hparams, 'train')
        valid_data = static_data_prep(hparams, 'valid')
        test_data = static_data_prep(hparams, 'test')
    else:
        raise NotImplementedError

    #  # Load pretrained model if pretrained_separator is present in the yaml
    #  if "pretrained_separator" in hparams:
    #      run_on_main(hparams["pretrained_separator"].collect_files)
    #      hparams["pretrained_separator"].load_collected()

    # load pretrained embedder
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()
    if isinstance(hparams["Embedder"], ECAPA_TDNN):
        hparams["Embedder"].eval()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    #  # re-initialize the parameters if we don't use a pretrained model
    #  if "pretrained_separator" not in hparams:
    #      for module in separator.modules.values():
    #          separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

    # Eval
    #  separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
