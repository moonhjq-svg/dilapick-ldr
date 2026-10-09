# Attribution and scope of rights

MIT is selected by the author for author-owned source and documentation in this
release. It does not replace upstream dataset terms or dependency licenses.

| Component | Relationship |
| --- | --- |
| PyTorch | Installed dependency; upstream BSD-style license, https://github.com/pytorch/pytorch/blob/main/LICENSE |
| NumPy | Installed dependency; upstream BSD-3-Clause, https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| STEAD | Dataset used for training/evaluation; raw waveforms excluded; obtain from https://github.com/smousavi05/STEAD under its terms |
| INSTANCE | External evaluation dataset; raw waveforms excluded; https://doi.org/10.5194/essd-13-5509-2021 |
| SeisBench | Dataset-format provenance; not vendored or required by the demo; https://github.com/seisbench/seisbench |
| Dense upsampling convolution / subpixel rearrangement | Prior mechanism acknowledged in the manuscript; no TuSimple source files are bundled |

PhaseNet, EQTransformer, LFTNet-inspired and LEQNet-V2B implementations and weights
are not bundled. No license grant is made for those external implementations.
Frozen split metadata retains dataset identifiers and arrival annotations; attribution
and applicable source-data terms remain in force. Checkpoint tensors and prediction
records are author-generated research artifacts provided with this release under
MIT to the extent of the authors' rights; upstream data rights are not relicensed.

The model files are the project's frozen internal implementations (see provenance),
not claimed official implementations of external models. An independent historical
authorship audit has not been performed; retain any upstream notice if additional
third-party derivation is later identified.
