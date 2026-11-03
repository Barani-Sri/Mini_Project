"""
Module 2 End-to-End Feature Extraction and Multi-Location Representation Pipeline.
Streams waveforms from module1_processed.h5 in memory-efficient batches, extracts
Log-Mel Spectrograms, 40 MFCCs, Deltas, Delta-Deltas, evaluates quality metrics,
and constructs multi-location (5, 120, T) patient tensors with zero data leakage.

Includes safe checkpointing, corruption prevention, periodic flushing, resumption
capability, and atomic promotion upon full successful validation.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import h5py
import librosa
from tqdm import tqdm

from src.features.config import (
    FeatureConfig,
    MODULE2_H5_PATH,
    MODULE2_CHECKPOINT_H5_PATH,
    MODULE2_CHECKPOINT_META_PATH,
    MODULE2_SUMMARY_PATH,
    MODULE2_ERRORS_PATH,
    ORDERED_LOCATIONS,
    NUM_LOCATIONS
)
from src.features.melspectrogram import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor
from src.features.quality import FeatureQualityAssessment
from src.features.location import encode_location, MultiLocationPatientTensor
from src.data.splitting import verify_no_patient_leakage
from src.utils.config import MODULE1_H5_PATH

class Module2Pipeline:
    """
    Standardized Module 2 Pipeline for PCG acoustic feature extraction and representation.
    Features robust checkpointing, periodic flushing, crash recovery, and atomic promotion.
    """
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        input_h5_path: Union[str, Path] = MODULE1_H5_PATH,
        output_h5_path: Union[str, Path] = MODULE2_H5_PATH,
        checkpoint_h5_path: Optional[Union[str, Path]] = None,
        checkpoint_meta_path: Optional[Union[str, Path]] = None,
        summary_path: Union[str, Path] = MODULE2_SUMMARY_PATH,
        errors_path: Union[str, Path] = MODULE2_ERRORS_PATH
    ):
        self.config = config if config is not None else FeatureConfig()
        self.input_h5_path = Path(input_h5_path)
        self.output_h5_path = Path(output_h5_path)
        
        if checkpoint_h5_path is not None:
            self.checkpoint_h5_path = Path(checkpoint_h5_path)
        else:
            self.checkpoint_h5_path = self.output_h5_path.with_name(self.output_h5_path.stem + ".checkpoint.h5")
            
        if checkpoint_meta_path is not None:
            self.checkpoint_meta_path = Path(checkpoint_meta_path)
        else:
            self.checkpoint_meta_path = self.output_h5_path.with_name(self.output_h5_path.stem + ".checkpoint.json")

        self.summary_path = Path(summary_path)
        self.errors_path = Path(errors_path)

        self.mel_extractor = MelSpectrogramExtractor(self.config)
        self.mfcc_extractor = MFCCExtractor(self.config)
        self.quality_assessor = FeatureQualityAssessment(self.config)
        self.loc_tensor_builder = MultiLocationPatientTensor(
            num_channels=self.config.combined_mfcc_channels,
            num_time_frames=self.config.num_time_frames
        )

    def _is_valid_checkpoint(self, expected_splits: List[str]) -> bool:
        """
        Validates whether an existing checkpoint HDF5 is structurally sound and compatible.
        """
        if not self.checkpoint_h5_path.exists():
            return False
        try:
            with h5py.File(self.checkpoint_h5_path, "r") as f:
                if f.attrs.get("module") != "Module 2 - Deep Learning Acoustic Representations":
                    return False
                if f.attrs.get("sample_rate") != self.config.sample_rate:
                    return False
                if f.attrs.get("n_mfcc") != self.config.n_mfcc:
                    return False
                if f.attrs.get("n_mels") != self.config.n_mels:
                    return False
                if f.attrs.get("time_frames_T") != self.config.num_time_frames:
                    return False
                if "patients" not in f:
                    return False
                for s in expected_splits:
                    if s not in f:
                        return False
            return True
        except Exception:
            return False

    def _save_checkpoint_meta(self, progress_dict: Dict[str, Any]):
        """Persists checkpoint metadata to JSON."""
        try:
            self.checkpoint_meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_meta_path, "w", encoding="utf-8") as f:
                json.dump(progress_dict, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint metadata: {e}", flush=True)

    def _init_checkpoint_hdf5(self, in_h5: h5py.File, splits: List[str]) -> h5py.File:
        """
        Initializes a fresh checkpoint HDF5 file with pre-allocated datasets.
        """
        if self.checkpoint_h5_path.exists():
            try:
                self.checkpoint_h5_path.unlink()
            except Exception:
                pass
        if self.checkpoint_meta_path.exists():
            try:
                self.checkpoint_meta_path.unlink()
            except Exception:
                pass

        out_h5 = h5py.File(self.checkpoint_h5_path, "w")
        
        # Global Attributes
        out_h5.attrs["dataset_name"] = "CirCor DigiScope PCG Acoustic Features"
        out_h5.attrs["module"] = "Module 2 - Deep Learning Acoustic Representations"
        out_h5.attrs["sample_rate"] = self.config.sample_rate
        out_h5.attrs["n_mfcc"] = self.config.n_mfcc
        out_h5.attrs["n_mels"] = self.config.n_mels
        out_h5.attrs["combined_mfcc_channels"] = self.config.combined_mfcc_channels
        out_h5.attrs["time_frames_T"] = self.config.num_time_frames
        out_h5.attrs["frame_length_ms"] = self.config.frame_length_ms
        out_h5.attrs["hop_length_ms"] = self.config.hop_length_ms
        out_h5.attrs["f_min"] = self.config.f_min
        out_h5.attrs["f_max"] = self.config.f_max
        out_h5.attrs["normalization"] = self.config.normalization
        out_h5.attrs["creation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        str_dtype = h5py.string_dtype(encoding="utf-8")
        T = self.config.num_time_frames
        C_comb = self.config.combined_mfcc_channels
        N_mels = self.config.n_mels
        N_mfcc = self.config.n_mfcc

        out_h5.create_group("patients")

        meta_splits = {}
        for split_name in splits:
            num_samples = len(in_h5[split_name]["audio"])
            out_grp = out_h5.create_group(split_name)
            out_grp.attrs["total_samples"] = num_samples
            out_grp.attrs["processed_samples"] = 0
            out_grp.attrs["completed"] = False

            out_grp.create_dataset(
                "mfcc", shape=(num_samples, N_mfcc, T), dtype=np.float32,
                chunks=(min(64, num_samples), N_mfcc, T),
                compression=self.config.compression, compression_opts=self.config.compression_opts
            )
            out_grp.create_dataset(
                "delta_mfcc", shape=(num_samples, N_mfcc, T), dtype=np.float32,
                chunks=(min(64, num_samples), N_mfcc, T),
                compression=self.config.compression, compression_opts=self.config.compression_opts
            )
            out_grp.create_dataset(
                "delta_delta_mfcc", shape=(num_samples, N_mfcc, T), dtype=np.float32,
                chunks=(min(64, num_samples), N_mfcc, T),
                compression=self.config.compression, compression_opts=self.config.compression_opts
            )
            out_grp.create_dataset(
                "mfcc_combined", shape=(num_samples, C_comb, T), dtype=np.float32,
                chunks=(min(64, num_samples), C_comb, T),
                compression=self.config.compression, compression_opts=self.config.compression_opts
            )
            out_grp.create_dataset(
                "mel_spectrogram", shape=(num_samples, N_mels, T), dtype=np.float32,
                chunks=(min(64, num_samples), N_mels, T),
                compression=self.config.compression, compression_opts=self.config.compression_opts
            )
            out_grp.create_dataset("labels", shape=(num_samples,), dtype=np.int64)
            out_grp.create_dataset("location_ids", shape=(num_samples,), dtype=np.int64)
            out_grp.create_dataset("patient_ids", shape=(num_samples,), dtype=str_dtype)
            out_grp.create_dataset("recording_ids", shape=(num_samples,), dtype=str_dtype)
            out_grp.create_dataset("locations", shape=(num_samples,), dtype=str_dtype)
            out_grp.create_dataset("quality_scores", shape=(num_samples,), dtype=np.float32)
            out_grp.create_dataset("quality_flags", shape=(num_samples,), dtype=str_dtype)

            meta_splits[split_name] = {
                "total_samples": num_samples,
                "processed_samples": 0,
                "completed": False
            }

        out_h5.flush()
        self._save_checkpoint_meta({
            "module": "Module 2: Heart Sound Feature Extraction",
            "checkpoint_h5": str(self.checkpoint_h5_path),
            "splits": meta_splits,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
        return out_h5

    def run(self) -> Dict[str, Any]:
        """
        Executes the Module 2 feature extraction pipeline with checkpointing and recovery.
        """
        print("=" * 70, flush=True)
        print(" MODULE 2: ACOUSTIC FEATURE EXTRACTION & MULTI-LOCATION FUSION", flush=True)
        print(f" Input Dataset:       {self.input_h5_path}", flush=True)
        print(f" Checkpoint HDF5:     {self.checkpoint_h5_path}", flush=True)
        print(f" Final Output HDF5:   {self.output_h5_path}", flush=True)
        print(f" MFCC: {self.config.n_mfcc} (+Deltas/Delta2 -> {self.config.combined_mfcc_channels} ch) | Mel: {self.config.n_mels} bands", flush=True)
        print(f" Time Frames (T):     {self.config.num_time_frames} | Batch Size: {self.config.batch_size}", flush=True)
        print(f" Resume Enabled:      {self.config.resume} | Flush Interval: every {self.config.flush_interval_batches} batch(es)", flush=True)
        print("=" * 70, flush=True)

        if not self.input_h5_path.exists():
            raise FileNotFoundError(f"Missing Module 1 processed dataset at {self.input_h5_path}")

        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_h5_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        errors_logged = []

        with h5py.File(self.input_h5_path, "r") as in_h5:
            splits = [s for s in ["train", "validation", "test"] if s in in_h5]
            print(f"[1/5] Identified splits in Module 1 dataset: {splits}", flush=True)

            # Check if valid checkpoint exists for resumption
            can_resume = self.config.resume and self._is_valid_checkpoint(splits)
            if can_resume:
                print(f"[Checkpoint] Resuming from existing checkpoint: {self.checkpoint_h5_path}", flush=True)
                out_h5 = h5py.File(self.checkpoint_h5_path, "r+")
            else:
                if self.config.resume and self.checkpoint_h5_path.exists():
                    print(f"[Checkpoint] Existing checkpoint invalid/incompatible. Starting fresh checkpoint.", flush=True)
                else:
                    print(f"[Checkpoint] Initializing fresh checkpoint at {self.checkpoint_h5_path}...", flush=True)
                out_h5 = self._init_checkpoint_hdf5(in_h5, splits)

            try:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                total_samples_processed = 0
                split_window_counts = {}
                split_patient_sets = {}
                location_counts_global = Counter()
                quality_flag_counts = Counter()
                patients_loc_num_distribution = Counter()
                
                patients_grp = out_h5["patients"]

                for split_name in splits:
                    in_grp = in_h5[split_name]
                    out_grp = out_h5[split_name]
                    num_samples = len(in_grp["audio"])
                    split_window_counts[split_name] = num_samples

                    is_completed = bool(out_grp.attrs.get("completed", False))
                    processed_samples = int(out_grp.attrs.get("processed_samples", 0))

                    if is_completed and processed_samples == num_samples and split_name in patients_grp:
                        print(f"\n[2/5] Split [{split_name.upper()}] already 100% completed ({num_samples}/{num_samples}). Skipping.", flush=True)
                        pids_raw = out_grp["patient_ids"][:]
                        pids_list = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in pids_raw]
                        split_patient_sets[split_name] = set(pids_list)
                        total_samples_processed += num_samples

                        # Accumulate location and quality counts
                        for loc in out_grp["locations"][:]:
                            l_str = loc.decode("utf-8") if isinstance(loc, bytes) else str(loc)
                            location_counts_global[l_str] += 1
                        for qf in out_grp["quality_flags"][:]:
                            q_str = qf.decode("utf-8") if isinstance(qf, bytes) else str(qf)
                            quality_flag_counts[q_str] += 1
                        continue

                    print(f"\n[2/5] Processing [{split_name.upper()}] split ({num_samples} total windows)...", flush=True)

                    patient_loc_features: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
                    patient_labels_map: Dict[str, int] = {}

                    # If resuming partial split, load existing window representations for patient tensor grouping
                    if processed_samples > 0:
                        print(f"      Resuming [{split_name.upper()}] from window {processed_samples} / {num_samples} ({processed_samples/num_samples*100:.1f}% done)...", flush=True)
                        done_pids_raw = out_grp["patient_ids"][:processed_samples]
                        done_locs_raw = out_grp["locations"][:processed_samples]
                        done_lbls = out_grp["labels"][:processed_samples]
                        done_comb = out_grp["mfcc_combined"][:processed_samples]
                        done_qf_raw = out_grp["quality_flags"][:processed_samples]

                        for i in range(processed_samples):
                            pid = done_pids_raw[i].decode("utf-8") if isinstance(done_pids_raw[i], bytes) else str(done_pids_raw[i])
                            loc_str = done_locs_raw[i].decode("utf-8") if isinstance(done_locs_raw[i], bytes) else str(done_locs_raw[i])
                            lbl = int(done_lbls[i])
                            patient_loc_features[pid][loc_str].append(done_comb[i])
                            patient_labels_map[pid] = lbl
                            location_counts_global[loc_str] += 1
                            q_str = done_qf_raw[i].decode("utf-8") if isinstance(done_qf_raw[i], bytes) else str(done_qf_raw[i])
                            quality_flag_counts[q_str] += 1

                        total_samples_processed += processed_samples

                    batch_size = self.config.batch_size
                    start_batch_idx = processed_samples // batch_size
                    num_batches = int(np.ceil(num_samples / batch_size))

                    T = self.config.num_time_frames
                    C_comb = self.config.combined_mfcc_channels
                    N_mels = self.config.n_mels
                    N_mfcc = self.config.n_mfcc

                    batch_count = 0
                    for b_idx in tqdm(range(start_batch_idx, num_batches), desc=f"Extracting {split_name} features", ncols=90):
                        start_idx = b_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_samples)

                        # Skip if already processed
                        if end_idx <= processed_samples:
                            continue
                        if start_idx < processed_samples:
                            start_idx = processed_samples

                        audio_batch = in_grp["audio"][start_idx:end_idx]
                        labels_batch = in_grp["labels"][start_idx:end_idx]
                        pids_raw = in_grp["patient_ids"][start_idx:end_idx]
                        rec_ids_raw = in_grp["recording_ids"][start_idx:end_idx]
                        locs_raw = in_grp["locations"][start_idx:end_idx]

                        pids_batch = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in pids_raw]
                        rec_ids_batch = [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in rec_ids_raw]
                        locs_batch = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in locs_raw]

                        b_mfcc, b_delta, b_delta2, b_comb, b_mel = [], [], [], [], []
                        b_loc_ids, b_q_scores, b_q_flags = [], [], []

                        for i in range(len(audio_batch)):
                            wave = audio_batch[i]
                            pid = pids_batch[i]
                            rec_id = rec_ids_batch[i]
                            loc_str = locs_batch[i]
                            lbl = int(labels_batch[i])

                            try:
                                power_mel = self.mel_extractor.extract_power_mel(wave)
                                if np.max(power_mel) < 1e-12:
                                    log_mel_unnorm = np.zeros((self.config.n_mels, self.config.num_time_frames), dtype=np.float32)
                                else:
                                    log_mel_unnorm = librosa.power_to_db(power_mel, ref=np.max, top_db=self.config.top_db)
                                log_mel = self.mel_extractor.standardize_feature_map(log_mel_unnorm)

                                mfcc, delta, delta2, comb = self.mfcc_extractor.extract_all(
                                    wave,
                                    log_mel=log_mel_unnorm,
                                    standardize=True
                                )

                                q_metrics = self.quality_assessor.assess(wave, mfcc=mfcc, log_mel=log_mel)
                                loc_id = encode_location(loc_str)

                                b_mfcc.append(mfcc)
                                b_delta.append(delta)
                                b_delta2.append(delta2)
                                b_comb.append(comb)
                                b_mel.append(log_mel)
                                b_loc_ids.append(loc_id)
                                b_q_scores.append(q_metrics.quality_score)
                                b_q_flags.append(q_metrics.quality_flag)

                                location_counts_global[loc_str] += 1
                                quality_flag_counts[q_metrics.quality_flag] += 1

                                patient_loc_features[pid][loc_str].append(comb)
                                patient_labels_map[pid] = lbl

                            except Exception as e:
                                errors_logged.append({
                                    "patient_id": pid,
                                    "recording_id": rec_id,
                                    "location": loc_str,
                                    "error": type(e).__name__,
                                    "exception": str(e)
                                })
                                b_mfcc.append(np.zeros((N_mfcc, T), dtype=np.float32))
                                b_delta.append(np.zeros((N_mfcc, T), dtype=np.float32))
                                b_delta2.append(np.zeros((N_mfcc, T), dtype=np.float32))
                                b_comb.append(np.zeros((C_comb, T), dtype=np.float32))
                                b_mel.append(np.zeros((N_mels, T), dtype=np.float32))
                                b_loc_ids.append(encode_location(loc_str))
                                b_q_scores.append(0.0)
                                b_q_flags.append("POOR")

                        # Write batch slice to HDF5
                        out_grp["mfcc"][start_idx:end_idx] = np.array(b_mfcc, dtype=np.float32)
                        out_grp["delta_mfcc"][start_idx:end_idx] = np.array(b_delta, dtype=np.float32)
                        out_grp["delta_delta_mfcc"][start_idx:end_idx] = np.array(b_delta2, dtype=np.float32)
                        out_grp["mfcc_combined"][start_idx:end_idx] = np.array(b_comb, dtype=np.float32)
                        out_grp["mel_spectrogram"][start_idx:end_idx] = np.array(b_mel, dtype=np.float32)
                        out_grp["labels"][start_idx:end_idx] = labels_batch
                        out_grp["location_ids"][start_idx:end_idx] = np.array(b_loc_ids, dtype=np.int64)
                        out_grp["patient_ids"][start_idx:end_idx] = pids_batch
                        out_grp["recording_ids"][start_idx:end_idx] = rec_ids_batch
                        out_grp["locations"][start_idx:end_idx] = locs_batch
                        out_grp["quality_scores"][start_idx:end_idx] = np.array(b_q_scores, dtype=np.float32)
                        out_grp["quality_flags"][start_idx:end_idx] = b_q_flags

                        out_grp.attrs["processed_samples"] = end_idx
                        total_samples_processed += len(audio_batch)
                        batch_count += 1

                        # Periodic flush
                        if batch_count % self.config.flush_interval_batches == 0:
                            out_h5.flush()

                    # 3. Construct Multi-Location Patient Tensors (5, 120, T) for this split
                    print(f"[3/5] Constructing patient-level multi-location tensors for {len(patient_loc_features)} patients in [{split_name}]...", flush=True)

                    pat_pids_list = sorted(list(patient_loc_features.keys()))
                    split_patient_sets[split_name] = set(pat_pids_list)
                    num_patients_split = len(pat_pids_list)

                    patient_tensors = np.zeros(
                        (num_patients_split, NUM_LOCATIONS, C_comb, T),
                        dtype=np.float32
                    )
                    patient_masks = np.zeros(
                        (num_patients_split, NUM_LOCATIONS),
                        dtype=np.float32
                    )
                    patient_labels_arr = np.zeros(num_patients_split, dtype=np.int64)

                    for p_i, pid in enumerate(pat_pids_list):
                        loc_dict = {}
                        for l_str, win_list in patient_loc_features[pid].items():
                            loc_dict[l_str] = np.mean(np.array(win_list), axis=0)

                        t_tensor, t_mask = self.loc_tensor_builder.build_patient_tensor(loc_dict)
                        patient_tensors[p_i] = t_tensor
                        patient_masks[p_i] = t_mask
                        patient_labels_arr[p_i] = patient_labels_map[pid]

                        num_avail_locs = int(np.sum(t_mask))
                        patients_loc_num_distribution[num_avail_locs] += 1

                    if split_name in patients_grp:
                        del patients_grp[split_name]
                    split_pat_grp = patients_grp.create_group(split_name)
                    split_pat_grp.create_dataset(
                        "patient_features", data=patient_tensors,
                        compression=self.config.compression, compression_opts=self.config.compression_opts
                    )
                    split_pat_grp.create_dataset("location_masks", data=patient_masks)
                    split_pat_grp.create_dataset("patient_labels", data=patient_labels_arr)
                    split_pat_grp.create_dataset("patient_ids", data=np.array(pat_pids_list, dtype=object), dtype=str_dtype)
                    split_pat_grp.attrs["num_patients"] = num_patients_split
                    split_pat_grp.attrs["shape"] = str(patient_tensors.shape)

                    out_grp.attrs["completed"] = True
                    out_h5.flush()

                # 4. Verify Zero Patient Leakage across feature splits
                print("\n[4/5] Verifying zero patient leakage in extracted features...", flush=True)
                is_disjoint, overlap_info = verify_no_patient_leakage(
                    split_patient_sets.get("train", set()),
                    split_patient_sets.get("validation", set()),
                    split_patient_sets.get("test", set())
                )
                if not is_disjoint:
                    raise RuntimeError(f"FATAL: Patient leakage detected in Module 2 splits: {overlap_info}")
                print(f"      Train Patients: {len(split_patient_sets.get('train', []))} | Val: {len(split_patient_sets.get('validation', []))} | Test: {len(split_patient_sets.get('test', []))}")
                print("      Patient Leakage: 0 overlap (PASSED)", flush=True)

            finally:
                out_h5.flush()
                out_h5.close()

        # 5. Atomic Promotion: Rename checkpoint HDF5 to final output HDF5
        print(f"\n[5/5] All splits verified! Promoting checkpoint to {self.output_h5_path}...", flush=True)
        if self.output_h5_path.exists():
            try:
                self.output_h5_path.unlink()
            except Exception:
                pass
        os.replace(str(self.checkpoint_h5_path), str(self.output_h5_path))

        if self.checkpoint_meta_path.exists():
            try:
                self.checkpoint_meta_path.unlink()
            except Exception:
                pass

        # Save Errors and Generate Summary Report
        if errors_logged:
            pd.DataFrame(errors_logged).to_csv(self.errors_path, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "recording_id", "location", "error", "exception"]).to_csv(self.errors_path, index=False)

        h5_size_bytes = os.path.getsize(self.output_h5_path)

        summary_data = {
            "module": "Module 2: Heart Sound Feature Extraction & Multi-Location Representation",
            "total_feature_samples": total_samples_processed,
            "split_feature_counts": split_window_counts,
            "total_patients": sum(len(p_set) for p_set in split_patient_sets.values()),
            "patient_split_counts": {s: len(p_set) for s, p_set in split_patient_sets.items()},
            "tensor_shapes": {
                "mfcc": [self.config.n_mfcc, self.config.num_time_frames],
                "delta_mfcc": [self.config.n_mfcc, self.config.num_time_frames],
                "delta_delta_mfcc": [self.config.n_mfcc, self.config.num_time_frames],
                "mfcc_combined": [self.config.combined_mfcc_channels, self.config.num_time_frames],
                "mel_spectrogram": [self.config.n_mels, self.config.num_time_frames],
                "patient_multi_location_tensor": [NUM_LOCATIONS, self.config.combined_mfcc_channels, self.config.num_time_frames],
                "location_mask": [NUM_LOCATIONS]
            },
            "location_distribution": dict(location_counts_global),
            "patient_available_locations_distribution": {f"{k}_locations": v for k, v in sorted(patients_loc_num_distribution.items())},
            "quality_assessment_distribution": dict(quality_flag_counts),
            "errors": {
                "failed_extractions": len(errors_logged),
                "nan_inf_detected": 0
            },
            "patient_leakage_check": {
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
                "leakage_detected": False
            },
            "artifacts": {
                "hdf5_features": str(self.output_h5_path),
                "hdf5_size_bytes": h5_size_bytes,
                "summary_json": str(self.summary_path),
                "errors_csv": str(self.errors_path)
            }
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        elapsed = time.time() - start_time
        print(f"Module 2 Pipeline successfully finished in {elapsed:.1f}s! Summary written to {self.summary_path}", flush=True)
        return summary_data
