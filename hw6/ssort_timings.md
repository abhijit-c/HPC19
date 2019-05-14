# Timings for Parallel Sample Sort

This is done with 64 processors requested from Prince. To compile and run the
ssort code, compile the file ssort.cpp and pass the argument $N$ to it in the
command line, which corresponds to the number of elements per processsor.

| Numel per Processor | Time (sec)        |
|---------------------|-------------------|
| 10000               | 0.734788 s        |
| 100000              | 0.742746 s        |
| 1000000             | 0.927943 s        |
