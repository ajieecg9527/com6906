"""
Recipe for training a Transformer ASR system with librispeech, from the SpeechBrain
LibriSpeech/ASR recipe. The SpeechBrain version used in this work is:
https://github.com/speechbrain/speechbrain/tree/1eddf66eea01866d3cf9dfe61b00bb48d2062236
"""

import sys

import speechbrain as sb
import torch
from fastdtw import fastdtw
from clarity.utils.file_io import read_jsonl
from scipy.spatial.distance import cosine
from speechbrain.utils.distributed import run_on_main

sys.path.append("../../")
from model.intelligibility.asr.decoder import S2STransformerBeamSearch


def dataio_prepare(csv_path, haspi_file, tokenizer, bos_index, eos_index):
    """ Data processing pipline """

    # 1. Load data
    dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=csv_path)
    datasets = [dataset]

    # Load haspi scores
    haspi_records = read_jsonl(haspi_file)
    better_ear_choices = {}
    for haspi_record in haspi_records:
        better_ear_choices[haspi_record["signal"]] = haspi_record["ear"]

    # 2. Define the signal pipeline
    @sb.utils.data_pipeline.takes("signal")
    @sb.utils.data_pipeline.provides("sig")
    def speech_pipeline(signal):
        sig = sb.dataio.dataio.read_audio(signal)
        # Choose the channel
        signal_name = signal.split("/")[-1].split(".")[0]
        ear = better_ear_choices[signal_name]
        return sig[:, ear]
    sb.dataio.dataset.add_dynamic_item(datasets, speech_pipeline)

    # 3. Define the text pipeline
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([bos_index] + tokens_list)
        yield tokens_bos  # begin of sentence
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos  # end of sentence
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Define the output
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"])

    return dataset


def dtw_similarity(x, y):
    path = fastdtw(x.detach().cpu().numpy()[0], y.detach().cpu().numpy()[0], dist=cosine)[1]

    x_, y_ = [], []
    for step in path:
        x_.append(x[:, step[0], :])
        y_.append(y[:, step[1], :])
    x_ = torch.stack(x_, dim=1)
    y_ = torch.stack(y_, dim=1)
    return torch.nn.functional.cosine_similarity(x_, y_, dim=-1)


def feature2similarity(signal_msbg_features, signal_ref_features, if_dtw=False):

    if if_dtw:
        similarity = dtw_similarity(signal_msbg_features, signal_ref_features)
        similarity = torch.max(torch.stack([torch.mean(similarity, dim=-1)], dim=-1), dim=-1)[0]
    else:
        max_len = torch.max(torch.LongTensor([signal_msbg_features.shape[1], signal_ref_features.shape[1]]))

        padded_signal_msbg_features = torch.zeros([1, max_len, signal_msbg_features.shape[2]], dtype=torch.float32)
        padded_signal_ref_features = torch.zeros([1, max_len, signal_ref_features.shape[2]], dtype=torch.float32)

        padded_signal_msbg_features[:, : signal_msbg_features.shape[1], :] = signal_msbg_features
        padded_signal_ref_features[:, : signal_ref_features.shape[1], :] = signal_ref_features

        similarity = torch.nn.functional.cosine_similarity(padded_signal_msbg_features, padded_signal_ref_features, dim=-1)
        similarity = torch.stack([similarity], dim=-1).max(dim=-1)[0]
        similarity = torch.mean(similarity, dim=-1)

    return similarity


def compute_similarity(sig_msbg, wrd, asr_model, bos_index, tokenizer, ear=0):
    len_sig = torch.tensor([1], dtype=torch.float32)  # relative length
    tokens_bos = torch.LongTensor([bos_index] + (tokenizer.encode_as_ids(wrd))).view(1, -1)

    sig_ref = sig_msbg.replace("msbg", "ref")

    # Monaural signal from better ear
    signal_msbg = sb.dataio.dataio.read_audio(sig_msbg)
    signal_msbg = signal_msbg[:, ear].view(1, -1)
    signal_ref = sb.dataio.dataio.read_audio(sig_ref)
    signal_ref = signal_ref[:, ear].view(1, -1)

    signal_msbg_features = asr_model.generate_features(signal_msbg, len_sig, tokens_bos)
    signal_ref_features = asr_model.generate_features(signal_ref, len_sig, tokens_bos)

    enc_similarity = feature2similarity(signal_msbg_features[0], signal_ref_features[0], if_dtw=False)
    dec_similarity = feature2similarity(signal_msbg_features[1], signal_ref_features[1], if_dtw=True)

    return enc_similarity[0].numpy(), dec_similarity[0].numpy()


class ASR(sb.core.Brain):
    """ An inherited class of the ASR model speech.core.brain,
        where the abstract methods compute_forward() and compute objectives() must be implemented.
        Plus, we define different training procedure for different stages."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_metric = None
        self.wer_metric = None
        self.train_stats = None
        self.switched = None
        self.optimizer = None
        self.tokenizer = None
        self.test_search = None

    def compute_forward(self, batch, stage):
        """ Forward pass from signals to word probabilities """
        batch = batch.to(self.device)
        sigs, len_sigs = batch.sig  # len_sigs is the relative length of the signals to the longest one
        tokens_bos, _ = batch.tokens_bos

        # Compute features
        features = self.hparams.compute_features(sigs)
        current_epoch = self.hparams.epoch_counter.current
        features = self.hparams.normalize(features, len_sigs, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                features = self.hparams.augmentation(features)

        # Forward pass
        src = self.hparams.CNN(features)
        enc_out, pred = self.hparams.Transformer(src, tokens_bos, len_sigs, pad_idx=self.hparams.pad_index)

        # Compute ctc log-probabilities
        logits = self.hparams.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # Compute seq2seq log-probabilities
        pred = self.hparams.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # For the sake of the efficiency, we only perform beams search with limited capacity
                # No language models to explain how the AM is working
                hyps, _ = self.hparams.valid_search(enc_out.detach(), len_sigs)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), len_sigs)

        return p_ctc, p_seq, len_sigs, hyps

    def compute_objectives(self, predictions, batch, stage):
        """ Compute the loss (CTC+NLL) given predictions and targets. """
        if stage != sb.Stage.TRAIN and (self.wer_metric is None or self.acc_metric is None):
            raise ValueError("wer_metric or acc_metric is None")
        (p_ctc, p_seq, len_sigs, hyps) = predictions

        ids = batch.id
        tokens_eos, len_tokens_eos = batch.tokens_eos
        tokens, len_tokens = batch.tokens

        loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, length=len_tokens_eos)
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, len_sigs, len_tokens)
        loss = self.hparams.ctc_weight * loss_ctc + (1 - self.hparams.ctc_weight) * loss_seq

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (stage == sb.Stage.TEST):
                # Decode token terms to words
                predicted_words = [self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            # Compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, len_tokens_eos)

        return loss

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def on_fit_start(self):
        """ Reinitialize the right optimizer at the start of the fitting """
        super().on_fit_start()

        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer

        # if stage two, then reinitialize the optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load the latest checkpoint to restart training if interrupted
            if self.checkpointer is not None:

                # Do not reload the weights if training is interrupted
                if "momentum" not in current_optimizer.param_groups[0]:
                    return

                # Recover the checkpointer if possible
                self.checkpointer.recover_if_possible(device=torch.device(self.device))

    def on_stage_start(self, stage, epoch=None):
        """ Get called at the start of each epoch """

        # Define the metrics for the val set and the test set
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def check_and_reset_optimizer(self):
        """ Reset the optimizer if stage 2 """
        current_epoch = self.hparams.epoch_counter.current
        if self.switched is None:
            self.switched = False

        if isinstance(self.optimizer, torch.optim.SGD):
            self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def fit_batch(self, batch):
        """ Train with one batch """

        # Switch optimizer from Adam to SGD, if stage 2
        if self.optimizer is None:
            raise ValueError("optimizer is None")
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # Normalize the loss
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # If the loss is not fini, then perform gradient clipping and early termination
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Anneal the learning rate every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """ Get called at the end of each epoch """
        if stage != sb.Stage.TRAIN and (self.wer_metric is None or self.acc_metric is None):
            raise ValueError("wer_metric or acc_metric is None")
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w", encoding="utf-8") as fp:
                self.wer_metric.write_stats(fp)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """ Perform checkpoint averge if needed """
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def evaluate_batch(self, batch, stage):
        """ Computations needed for validation/test batch """

        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)

        return loss.detach()

    def init_evaluation(self, max_key=None, min_key=None):
        """perform checkpoint averege if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )
        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

        self.test_search = S2STransformerBeamSearch(
            modules=[
                self.hparams.Transformer,
                self.hparams.seq_lin,
                self.hparams.ctc_lin,
            ],
            bos_index=self.hparams.bos_index,
            eos_index=self.hparams.eos_index,
            blank_index=self.hparams.blank_index,
            min_decode_ratio=self.hparams.min_decode_ratio,
            max_decode_ratio=self.hparams.max_decode_ratio,
            beam_size=self.hparams.test_beam_size,
            ctc_weight=self.hparams.ctc_weight_decode,
            lm_weight=self.hparams.lm_weight,
            lm_modules=self.hparams.lm_model,
            temperature=1,
            temperature_lm=1,
            topk=10,
            using_eos_threshold=False,
            length_normalization=True,
        )

    def generate_features(self, sigs, len_sigs, tokens_bos):
        """Forward computations from the waveform batches to the output probs."""
        # batch = batch.to(self.device)
        if self.test_search is None:
            raise ValueError("test_search is not initialized")

        sigs, len_sigs, tokens_bos = (
            sigs.to(self.device),
            len_sigs.to(self.device),
            tokens_bos.to(self.device),
        )
        with torch.no_grad():
            features = self.hparams.compute_features(sigs)
            current_epoch = self.hparams.epoch_counter.current
            feats = self.hparams.normalize(features, len_sigs, epoch=current_epoch)

            cnn_out = self.hparams.CNN(feats)
            enc_out, _ = self.hparams.Transformer(
                cnn_out, tokens_bos, len_sigs, pad_idx=self.hparams.pad_index
            )
            _, _, dec_out, _ = self.test_search(enc_out.detach(), len_sigs)

        return enc_out.detach().cpu(), dec_out.unsqueeze(0).detach().cpu()
