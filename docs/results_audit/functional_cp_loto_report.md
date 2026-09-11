# SAFE Functional CP Under LOTO

- Alphas: `0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3`
- CP seeds: `0,1,2,3,4,5,6,7,8,9`
- Horizon: `520`
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
