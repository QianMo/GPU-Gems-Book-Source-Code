GPU-Based Option Pricing Demo
-----------------------------

This command-line application demonstrates how two different option
pricing models -- Black-Scholes and binomial lattice -- can be
implemented on the GPU.

Requirements
------------

To run the demo, you will need:

- PC running windows 2000/XP
- GPU capable of running moderately complex fragment programs

To recompile the demo, you will need:

- Cg Toolkit version 1.2 or later (see developer.nvidia.com)

Running
-------

The application generates random input data (strike price, time
to expiration, volatility, etc.) for each of the specified number of
options, and then prices the options using the specified method.
By default, the options are priced using both the CPU and GPU,
with timing results and difference statistics printed to the console.

You must run the application from the top-directory containing
the various .cg source files.

usage: options [arguments]

Where arguments consists of:

    -put        Price puts.
    -call       Price calls.
    -n nopts    Price nopts options.
    -sn nopts   Price nopts^2 options.
    -a          Assume American-style exercise.
    -e          Assume European-style exercise. 
    -bs         Price using Black-Scholes.
    -bin nsteps Price using binomial lattice model with nsteps steps.
    -t ntests   Price each set of option ntests times.
    -cpu        Don't run CPU-based pricing.
    -gpu        Don't run GPU-based pricing.
    -rb         Don't read back results from GPU. 
    -seed seed  Set random number seed.
    -h          Print help.


Examples
--------

2048 American put options priced using binomial lattice model with 512 steps,
repeated 10 times:

% options -a -put -bin 512 -n 2048 -t 10
Initialization: 0.214s
2048 American put options, Binomial (512 steps)
CPU Elapsed time for 10 runs: 26.151 (0.783 K/sec)
GPU Elapsed time for 10 runs: 4.567 (4.485 K/sec)
RMS diff: 6.75417e-005, Max diff: 0.035%, GPU Avg: 7.45583, CPU Avg: 7.45588


1024*1024 European call options priced using Black-Scholes, repeated 20 times:

% options -e -call -bs -sn 1024 -t 20
Initialization: 0.328s
1048576 European call options, Black-Scholes
CPU Elapsed time for 20 runs: 7.208 (2909.319 K/sec)
GPU Elapsed time for 20 runs: 1.160 (18074.157 K/sec)
RMS diff: 1.494e-006, Max diff: 0.027%, GPU Avg: 3.909, CPU Avg: 3.909
