---
title: Homework 3
author: Abhijit Chowdhary
date: \today{}
geometry: margin=2cm
fontsize: 12pt
header-includes:
    - \usepackage{amsmath}
---

# Parallel Scan
I'm running this on NYU's Dumbo0 server, which has 48 cores avaliable
corresponding to a ``Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz``.

| Number of Cores | Runtime  |
|-----------------|----------|
| 1               | .250261s |
| 3               | .122880s |
| 6               | .073091s |
| 9               | .057125s |
| 12              | .051304s |
| 15              | .048222s |
| 18              | .049743s |
| 21              | .053330s |
| 24              | .054989s |
| 48              | .060081s |

So we notice something interesting here, that after about 12 to 15 cores, the
speedup we recieve is negligible. Initially, I thought this was strange,
however I think this is likely because the problem is very in computational
intensity. We have a extremely large array, and we're just summing across, not
many computations per memory access. Actually, when I added a bunch of useless
work into the scan sequential code, the disparity between the sequential and the
parallel versions widened.

![Plot of timings.](./timings.pdf)
