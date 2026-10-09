# Publishing this prepared release

Upload this directory or its complete ZIP to the chosen permanent repository.
No server credentials, waveforms, optimizer caches, Jetson engines or manuscript
source are required. Keep `weights/`, `data/`, `configs/`, `evidence/`, and
`SHA256SUMS.json` with the code. Verify a freshly downloaded copy using the README.

Do not describe the code as public until the actual release URL is accessible.
The manuscript statement can then describe its verified scope, for example:

> Model definitions, frozen checkpoints and data splits, inference code, and scripts
> for reproducing the primary recovery-comparison table from frozen predictions are
> available at [actual release URL]. Full retraining has not been independently
> validated in a clean environment.

The bracketed URL above is an editorial template, not a live link or a statement
already inserted into the manuscript. If submitting before publication, describe
the actual sharing mechanism agreed with the corresponding author.

GRSL's official author checklist section 5.4 encourages public source code and
sufficient detail on code, splits, experiment settings and relevant environments:
https://www.grss-ieee.org/publications/checklist-for-authors/
