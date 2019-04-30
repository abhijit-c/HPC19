---
title: Homework 5
author: Abhijit Chowdhary
date: \today{}
geometry: margin=2cm
fontsize: 12pt
header-includes:
    - \usepackage{amsmath}
---

# MPI Ring Communication

See the file `int_ring.cpp` for the single element ring and see the file
`arr_ring.cpp` for the array of size 2MB ring. Running this on an Intel(R)
Core(TM) i7-3770 CPU @ 3.40GHz, I found that on average my latency was around
.0035 ms and had bandwidth 10 GB/s on my local computer. You can call the shell
script provided to automatically run both codes. For example, call `sh
run_all.sh N` to run the files with N processors.

# Proposed Schedule

* Week 4/22-4/28: 
    - Read papers and understand basic Parareal.
    - Begin writing basic serial version in Matlab for correctness (almost
        done).
* Week 4/29-5/05:
    - Finish writing Serial version, and port code to C++.
    - Write data structures and helper code for ODE systems and solvers.
    - Perform numerical analysis on convergence, stability, and robustness of
        methods as internal solvers vary.
* Week 5/06-5/12:
    - Take serial version and write basic OpenMP version.
        - Compare efficiency.
        - Model computational intensity, and think about how to maximize it.
    - Begin seriously writing and wrapping up an MPI version to scale to larger
        systems.
    - Once this is done, try it on Prince.
* Week 5/13-5/19:
    - Construct plots and figures for both report and presentation.
    - Finalize mathematics behind optimizing parallel efficiency and
        computational intensity. Confirm with professors about they accuracy.
    - Finish typesetting both.
