#!/usr/bin/env bash
# Regenerates the six figures that need trained models, in the shared style. Output: figs_new/
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p figs_new
python scripts/plot_teaser.py       --dataset Pre-Pro                               --out figs_new/fig1_teaser_new.pdf
python scripts/plot_density.py      --datasets volcano earthquake flood fire        --out figs_new/fig_earth_density.pdf
python scripts/plot_density.py      --datasets Glycine Proline Pre-Pro              --out figs_new/fig_rama.pdf
python scripts/plot_displacement.py --dataset Pre-Pro                               --out figs_new/fig_displacement.pdf
python scripts/plot_samples.py      --datasets fire cone fisher24 swiss             --out figs_new/fig6_samples.pdf
python scripts/plot_samples.py      --datasets cone fisher24 line swiss --style 3d  --out figs_new/fig6_so3_3d.pdf
ls -la figs_new/*.pdf
