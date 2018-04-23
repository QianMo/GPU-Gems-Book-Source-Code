Command-line parameters:
- first parameter is an integer indicating the number of nodes
  (atoms).  Use powers of 2 for efficient pbuffers
- second parameter is an integer indicating the number of times
  the computation is repeated to average timing results.  If 0
  (zero) is specified, the experiment is executed once and the
  results (distance matrices) are written to the screen.  The
  results are not written for any other numbers of experiments.

If no command-line parameters are specified then the number
of nodes is set to 8, experiment is executed once, and the 
results are written to the console.  If only the number of nodes 
is specified, the experiment is executed once and the results are
written to the screen.

Examples:

gpu-bounds.exe 512 10  - time bound-smoothing for 512 atoms,
			 average results over 10 runs

gpu-bounds.exe 8 0     - perform bound-smoothing for 16 atoms once,
			 write the updated bounds to the console

gpu-bounds.exe	       - same as gpu-bounds.exe 8 0
