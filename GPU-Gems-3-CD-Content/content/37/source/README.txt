Unfortunately, this code only works in linux at this point in time. It may, of course, be possible to get it working under cygwin.

Updated versions of the code are likely to be available from:
http://www.doc.ic.ac.uk/~lwh01/perm/gpugems3

1) Set paths appropriately in the makefile
2) Ensure that the GNU Scientific library (GSL) is installed. This is available from http://www.gnu.org/software/gsl/ or from your local package management source. Alternatively it is available in source form in the gsl subdirectory.
3) Type make.
4) Assuming there are no errors... type ./test
5) Watch the simulation run. Timing results are displayed for various parts of the execution, followed by the results for each of the two CUDA random number generators as run through a basic chi squared test. Only a basic test is included as full tests have extended execution time.

