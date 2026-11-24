# PyMOL script: render the fragment ensembles of one example (run from figs_new/fragments_pdb)
#   pymol -cq render_fragments.pml
load example1_real.pdb, real
load example1_uniform.pdb, uniform
load example1_gmm.pdb, statistical
hide everything
set all_states, on
show sticks, name N+CA+C
set stick_radius, 0.12
color grey50, uniform
color skyblue, statistical
color black, real
set stick_radius, 0.3, real
bg_color white
set ray_opaque_background, 0
orient real
disable statistical
png example1_uniform.png, width=1600, height=1200, dpi=300, ray=1
disable uniform
enable statistical
png example1_statistical.png, width=1600, height=1200, dpi=300, ray=1
