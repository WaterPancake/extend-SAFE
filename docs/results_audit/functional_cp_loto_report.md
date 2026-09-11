# SAFE Functional CP Under LOTO

- Alphas: `0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3`
- CP seeds: `0,1,2,3,4,5,6,7,8,9`
- Horizon: `520`
- Fixed horizons: `50,100,148`
- Calibration: successful seen-task validation rollouts only
- Uncertainty: CP-seed mean within task, then task-macro; 95% task-bootstrap CI

## Macro tradeoff

| model | eval | alpha | realized FPR | catch | caught alarm frac | effective alarm frac |
|---|---|---:|---:|---:|---:|---:|
| lstm | by earliest stop | 0.01 | 0.038 | 0.069 | 0.491 | 0.957 |
| lstm | by earliest stop | 0.025 | 0.063 | 0.136 | 0.531 | 0.922 |
| lstm | by earliest stop | 0.05 | 0.103 | 0.227 | 0.669 | 0.906 |
| lstm | by earliest stop | 0.075 | 0.144 | 0.276 | 0.627 | 0.870 |
| lstm | by earliest stop | 0.1 | 0.184 | 0.344 | 0.607 | 0.843 |
| lstm | by earliest stop | 0.15 | 0.273 | 0.451 | 0.603 | 0.793 |
| lstm | by earliest stop | 0.2 | 0.375 | 0.567 | 0.577 | 0.727 |
| lstm | by earliest stop | 0.3 | 0.514 | 0.692 | 0.460 | 0.604 |
| lstm | by final end | 0.01 | 0.044 | 0.175 | 0.432 | 0.918 |
| lstm | by final end | 0.025 | 0.075 | 0.310 | 0.471 | 0.842 |
| lstm | by final end | 0.05 | 0.129 | 0.457 | 0.433 | 0.756 |
| lstm | by final end | 0.075 | 0.190 | 0.570 | 0.414 | 0.684 |
| lstm | by final end | 0.1 | 0.245 | 0.642 | 0.392 | 0.623 |
| lstm | by final end | 0.15 | 0.336 | 0.760 | 0.382 | 0.531 |
| lstm | by final end | 0.2 | 0.434 | 0.810 | 0.329 | 0.454 |
| lstm | by final end | 0.3 | 0.566 | 0.883 | 0.273 | 0.350 |
| lstm | fixed horizon 100 | 0.01 | 0.052 | 0.067 | 0.294 | 0.944 |
| lstm | fixed horizon 100 | 0.025 | 0.083 | 0.111 | 0.336 | 0.911 |
| lstm | fixed horizon 100 | 0.05 | 0.122 | 0.170 | 0.469 | 0.875 |
| lstm | fixed horizon 100 | 0.075 | 0.166 | 0.227 | 0.437 | 0.842 |
| lstm | fixed horizon 100 | 0.1 | 0.202 | 0.273 | 0.471 | 0.830 |
| lstm | fixed horizon 100 | 0.15 | 0.271 | 0.359 | 0.475 | 0.793 |
| lstm | fixed horizon 100 | 0.2 | 0.349 | 0.460 | 0.473 | 0.730 |
| lstm | fixed horizon 100 | 0.3 | 0.498 | 0.625 | 0.378 | 0.581 |
| lstm | fixed horizon 148 | 0.01 | 0.049 | 0.070 | 0.314 | 0.945 |
| lstm | fixed horizon 148 | 0.025 | 0.078 | 0.133 | 0.423 | 0.907 |
| lstm | fixed horizon 148 | 0.05 | 0.119 | 0.205 | 0.540 | 0.878 |
| lstm | fixed horizon 148 | 0.075 | 0.176 | 0.295 | 0.568 | 0.844 |
| lstm | fixed horizon 148 | 0.1 | 0.225 | 0.361 | 0.587 | 0.830 |
| lstm | fixed horizon 148 | 0.15 | 0.311 | 0.456 | 0.531 | 0.767 |
| lstm | fixed horizon 148 | 0.2 | 0.408 | 0.567 | 0.494 | 0.696 |
| lstm | fixed horizon 148 | 0.3 | 0.550 | 0.712 | 0.404 | 0.564 |
| lstm | fixed horizon 50 | 0.01 | 0.054 | 0.069 | 0.277 | 0.945 |
| lstm | fixed horizon 50 | 0.025 | 0.081 | 0.114 | 0.268 | 0.908 |
| lstm | fixed horizon 50 | 0.05 | 0.111 | 0.154 | 0.292 | 0.877 |
| lstm | fixed horizon 50 | 0.075 | 0.138 | 0.186 | 0.241 | 0.849 |
| lstm | fixed horizon 50 | 0.1 | 0.161 | 0.215 | 0.241 | 0.828 |
| lstm | fixed horizon 50 | 0.15 | 0.216 | 0.282 | 0.268 | 0.786 |
| lstm | fixed horizon 50 | 0.2 | 0.292 | 0.372 | 0.260 | 0.721 |
| lstm | fixed horizon 50 | 0.3 | 0.446 | 0.553 | 0.166 | 0.550 |
| mlp | by earliest stop | 0.01 | 0.003 | 0.011 | 0.767 | 0.996 |
| mlp | by earliest stop | 0.025 | 0.006 | 0.020 | 0.674 | 0.990 |
| mlp | by earliest stop | 0.05 | 0.032 | 0.041 | 0.522 | 0.974 |
| mlp | by earliest stop | 0.075 | 0.048 | 0.085 | 0.452 | 0.955 |
| mlp | by earliest stop | 0.1 | 0.108 | 0.135 | 0.341 | 0.933 |
| mlp | by earliest stop | 0.15 | 0.139 | 0.165 | 0.404 | 0.923 |
| mlp | by earliest stop | 0.2 | 0.175 | 0.224 | 0.341 | 0.876 |
| mlp | by earliest stop | 0.3 | 0.372 | 0.412 | 0.186 | 0.674 |
| mlp | by final end | 0.01 | 0.013 | 0.167 | 0.917 | 0.981 |
| mlp | by final end | 0.025 | 0.026 | 0.238 | 0.884 | 0.962 |
| mlp | by final end | 0.05 | 0.089 | 0.520 | 0.846 | 0.896 |
| mlp | by final end | 0.075 | 0.144 | 0.718 | 0.791 | 0.826 |
| mlp | by final end | 0.1 | 0.199 | 0.822 | 0.730 | 0.758 |
| mlp | by final end | 0.15 | 0.272 | 0.938 | 0.647 | 0.661 |
| mlp | by final end | 0.2 | 0.329 | 0.953 | 0.575 | 0.589 |
| mlp | by final end | 0.3 | 0.439 | 0.967 | 0.431 | 0.442 |
| mlp | fixed horizon 100 | 0.01 | 0.017 | 0.017 | 0.283 | 0.989 |
| mlp | fixed horizon 100 | 0.025 | 0.038 | 0.035 | 0.247 | 0.974 |
| mlp | fixed horizon 100 | 0.05 | 0.051 | 0.063 | 0.480 | 0.955 |
| mlp | fixed horizon 100 | 0.075 | 0.057 | 0.083 | 0.405 | 0.950 |
| mlp | fixed horizon 100 | 0.1 | 0.079 | 0.101 | 0.450 | 0.946 |
| mlp | fixed horizon 100 | 0.15 | 0.110 | 0.160 | 0.467 | 0.919 |
| mlp | fixed horizon 100 | 0.2 | 0.221 | 0.271 | 0.440 | 0.850 |
| mlp | fixed horizon 100 | 0.3 | 0.390 | 0.425 | 0.313 | 0.674 |
| mlp | fixed horizon 148 | 0.01 | 0.016 | 0.019 | 0.377 | 0.988 |
| mlp | fixed horizon 148 | 0.025 | 0.030 | 0.037 | 0.564 | 0.974 |
| mlp | fixed horizon 148 | 0.05 | 0.048 | 0.068 | 0.543 | 0.960 |
| mlp | fixed horizon 148 | 0.075 | 0.062 | 0.088 | 0.506 | 0.951 |
| mlp | fixed horizon 148 | 0.1 | 0.078 | 0.111 | 0.431 | 0.936 |
| mlp | fixed horizon 148 | 0.15 | 0.132 | 0.172 | 0.465 | 0.907 |
| mlp | fixed horizon 148 | 0.2 | 0.215 | 0.280 | 0.422 | 0.844 |
| mlp | fixed horizon 148 | 0.3 | 0.396 | 0.447 | 0.335 | 0.662 |
| mlp | fixed horizon 50 | 0.01 | 0.021 | 0.018 | 0.199 | 0.988 |
| mlp | fixed horizon 50 | 0.025 | 0.039 | 0.036 | 0.184 | 0.973 |
| mlp | fixed horizon 50 | 0.05 | 0.060 | 0.067 | 0.304 | 0.952 |
| mlp | fixed horizon 50 | 0.075 | 0.070 | 0.088 | 0.422 | 0.941 |
| mlp | fixed horizon 50 | 0.1 | 0.080 | 0.101 | 0.380 | 0.936 |
| mlp | fixed horizon 50 | 0.15 | 0.124 | 0.157 | 0.445 | 0.896 |
| mlp | fixed horizon 50 | 0.2 | 0.182 | 0.225 | 0.412 | 0.842 |
| mlp | fixed horizon 50 | 0.3 | 0.373 | 0.401 | 0.275 | 0.685 |

## Constraint-selected operating points

| model | eval | constraint | met | alpha | FPR | catch | effective alarm frac |
|---|---|---|---|---:|---:|---:|---:|
| lstm | by earliest stop | FPR <= 1% | False | 0.01 | 0.038 | 0.069 | 0.957 |
| lstm | by earliest stop | FPR <= 5% | True | 0.01 | 0.038 | 0.069 | 0.957 |
| lstm | by earliest stop | FPR <= 10% | True | 0.025 | 0.063 | 0.136 | 0.922 |
| lstm | by earliest stop | catch >= 25% | True | 0.075 | 0.144 | 0.276 | 0.870 |
| lstm | by earliest stop | catch >= 50% | True | 0.2 | 0.375 | 0.567 | 0.727 |
| lstm | by earliest stop | catch >= 75% | False | 0.3 | 0.514 | 0.692 | 0.604 |
| lstm | by final end | FPR <= 1% | False | 0.01 | 0.044 | 0.175 | 0.918 |
| lstm | by final end | FPR <= 5% | True | 0.01 | 0.044 | 0.175 | 0.918 |
| lstm | by final end | FPR <= 10% | True | 0.025 | 0.075 | 0.310 | 0.842 |
| lstm | by final end | catch >= 25% | True | 0.025 | 0.075 | 0.310 | 0.842 |
| lstm | by final end | catch >= 50% | True | 0.075 | 0.190 | 0.570 | 0.684 |
| lstm | by final end | catch >= 75% | True | 0.15 | 0.336 | 0.760 | 0.531 |
| lstm | fixed horizon 100 | FPR <= 1% | False | 0.01 | 0.052 | 0.067 | 0.944 |
| lstm | fixed horizon 100 | FPR <= 5% | False | 0.01 | 0.052 | 0.067 | 0.944 |
| lstm | fixed horizon 100 | FPR <= 10% | True | 0.025 | 0.083 | 0.111 | 0.911 |
| lstm | fixed horizon 100 | catch >= 25% | True | 0.1 | 0.202 | 0.273 | 0.830 |
| lstm | fixed horizon 100 | catch >= 50% | True | 0.3 | 0.498 | 0.625 | 0.581 |
| lstm | fixed horizon 100 | catch >= 75% | False | 0.3 | 0.498 | 0.625 | 0.581 |
| lstm | fixed horizon 148 | FPR <= 1% | False | 0.01 | 0.049 | 0.070 | 0.945 |
| lstm | fixed horizon 148 | FPR <= 5% | True | 0.01 | 0.049 | 0.070 | 0.945 |
| lstm | fixed horizon 148 | FPR <= 10% | True | 0.025 | 0.078 | 0.133 | 0.907 |
| lstm | fixed horizon 148 | catch >= 25% | True | 0.075 | 0.176 | 0.295 | 0.844 |
| lstm | fixed horizon 148 | catch >= 50% | True | 0.2 | 0.408 | 0.567 | 0.696 |
| lstm | fixed horizon 148 | catch >= 75% | False | 0.3 | 0.550 | 0.712 | 0.564 |
| lstm | fixed horizon 50 | FPR <= 1% | False | 0.01 | 0.054 | 0.069 | 0.945 |
| lstm | fixed horizon 50 | FPR <= 5% | False | 0.01 | 0.054 | 0.069 | 0.945 |
| lstm | fixed horizon 50 | FPR <= 10% | True | 0.025 | 0.081 | 0.114 | 0.908 |
| lstm | fixed horizon 50 | catch >= 25% | True | 0.15 | 0.216 | 0.282 | 0.786 |
| lstm | fixed horizon 50 | catch >= 50% | True | 0.3 | 0.446 | 0.553 | 0.550 |
| lstm | fixed horizon 50 | catch >= 75% | False | 0.3 | 0.446 | 0.553 | 0.550 |
| mlp | by earliest stop | FPR <= 1% | True | 0.025 | 0.006 | 0.020 | 0.990 |
| mlp | by earliest stop | FPR <= 5% | True | 0.075 | 0.048 | 0.085 | 0.955 |
| mlp | by earliest stop | FPR <= 10% | True | 0.075 | 0.048 | 0.085 | 0.955 |
| mlp | by earliest stop | catch >= 25% | True | 0.3 | 0.372 | 0.412 | 0.674 |
| mlp | by earliest stop | catch >= 50% | False | 0.3 | 0.372 | 0.412 | 0.674 |
| mlp | by earliest stop | catch >= 75% | False | 0.3 | 0.372 | 0.412 | 0.674 |
| mlp | by final end | FPR <= 1% | False | 0.01 | 0.013 | 0.167 | 0.981 |
| mlp | by final end | FPR <= 5% | True | 0.025 | 0.026 | 0.238 | 0.962 |
| mlp | by final end | FPR <= 10% | True | 0.05 | 0.089 | 0.520 | 0.896 |
| mlp | by final end | catch >= 25% | True | 0.05 | 0.089 | 0.520 | 0.896 |
| mlp | by final end | catch >= 50% | True | 0.05 | 0.089 | 0.520 | 0.896 |
| mlp | by final end | catch >= 75% | True | 0.1 | 0.199 | 0.822 | 0.758 |
| mlp | fixed horizon 100 | FPR <= 1% | False | 0.01 | 0.017 | 0.017 | 0.989 |
| mlp | fixed horizon 100 | FPR <= 5% | True | 0.025 | 0.038 | 0.035 | 0.974 |
| mlp | fixed horizon 100 | FPR <= 10% | True | 0.1 | 0.079 | 0.101 | 0.946 |
| mlp | fixed horizon 100 | catch >= 25% | True | 0.2 | 0.221 | 0.271 | 0.850 |
| mlp | fixed horizon 100 | catch >= 50% | False | 0.3 | 0.390 | 0.425 | 0.674 |
| mlp | fixed horizon 100 | catch >= 75% | False | 0.3 | 0.390 | 0.425 | 0.674 |
| mlp | fixed horizon 148 | FPR <= 1% | False | 0.01 | 0.016 | 0.019 | 0.988 |
| mlp | fixed horizon 148 | FPR <= 5% | True | 0.05 | 0.048 | 0.068 | 0.960 |
| mlp | fixed horizon 148 | FPR <= 10% | True | 0.1 | 0.078 | 0.111 | 0.936 |
| mlp | fixed horizon 148 | catch >= 25% | True | 0.2 | 0.215 | 0.280 | 0.844 |
| mlp | fixed horizon 148 | catch >= 50% | False | 0.3 | 0.396 | 0.447 | 0.662 |
| mlp | fixed horizon 148 | catch >= 75% | False | 0.3 | 0.396 | 0.447 | 0.662 |
| mlp | fixed horizon 50 | FPR <= 1% | False | 0.01 | 0.021 | 0.018 | 0.988 |
| mlp | fixed horizon 50 | FPR <= 5% | True | 0.025 | 0.039 | 0.036 | 0.973 |
| mlp | fixed horizon 50 | FPR <= 10% | True | 0.1 | 0.080 | 0.101 | 0.936 |
| mlp | fixed horizon 50 | catch >= 25% | True | 0.3 | 0.373 | 0.401 | 0.685 |
| mlp | fixed horizon 50 | catch >= 50% | False | 0.3 | 0.373 | 0.401 | 0.685 |
| mlp | fixed horizon 50 | catch >= 75% | False | 0.3 | 0.373 | 0.401 | 0.685 |
