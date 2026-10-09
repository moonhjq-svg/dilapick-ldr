# Data acquisition and frozen experiment

Raw STEAD and INSTANCE waveforms are not redistributed.

- STEAD: https://github.com/smousavi05/STEAD (Mousavi et al., 2019).
- INSTANCE: https://doi.org/10.5194/essd-13-5509-2021 (Michelini et al., 2021).
- The paper used a SeisBench-format STEAD mirror. The supplied trace identifiers
  such as `bucket817$361,:3,:6000` refer to that bucket layout. They are not original
  STEAD HDF5 dataset names. Use the corresponding SeisBench metadata and waveform
  release to resolve them; simply pointing the historical loader at the original
  STEAD download will not reproduce this experiment.

The complete frozen index is `data/stead_50k_50k_index.csv`, SHA256
`322760be43d0d901472fde2b8165e5bc94acc392f4eafbc647a76303c9440ef9`.
Use its explicit `split` column and order, never generate a new random split.
There are 40,000 event + 40,000 noise train records, 5,000 + 5,000 validation
records, and 5,000 + 5,000 test records. Sampling seed: 20260630.
This is a record-level stratified split. It is not source- or station-disjoint:
615 test event records share a source with train or validation. See
`evidence/split_audit.json`. Waveform-content duplication was not assessed.

Inputs are three-component Z/N/E windows at 100 Hz, 6000 samples. Normalize each
channel independently by subtracting its mean and dividing by population standard
deviation with a `1e-6` floor. P/S training targets are Gaussian with sigma 10
samples; Detection spans P minus 100 to S plus 500 samples, clipped to the window.
Historical label generation and input loading sources are retained in
`reference_training/`. Training-label integer handling must be read from these
sources; the published rescoring separately canonicalizes test reference arrivals
by rounding original metadata for every model.

This release does not provide a validated end-to-end waveform preparation or
retraining command. It provides a fully executable raw-array inference entry point
and frozen-prediction rescoring. No INSTANCE waveform download is required for these
two commands; INSTANCE experiments remain in the separate full evidence package.
