# Results

Test requests: 1,688,311. Fixed shading k tuned on validation: 0.27; hindsight-best k on test: 0.27. Dropped features: weekday.

`fixed_shading_hindsight` picks k on the test set itself: an unfair upper bound for the fixed baseline.

## Strategies (first-price simulation on test)

| Strategy | Surplus, CNY | vs fixed | vs hindsight fixed | Win rate | Spend, CNY | Avg paid CPM, fen | Share of oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| truthful | 0.00 | -100.0% | -100.0% | 100.0% | 5,064.93 | 300.0 | 0.0% |
| fixed_shading | 2,828.46 | 0.0% | 0.0% | 76.5% | 1,046.14 | 81.0 | 74.0% |
| fixed_shading_hindsight | 2,828.46 | 0.0% | 0.0% | 76.5% | 1,046.14 | 81.0 | 74.0% |
| shading_km_global | 2,826.57 | -0.1% | -0.1% | 77.5% | 1,099.22 | 84.0 | 74.0% |
| shading_km_segmented | 2,892.70 | 2.3% | 2.3% | 82.2% | 1,269.33 | 91.5 | 75.7% |
| shading_lgbm_win_classifier | 3,023.67 | 6.9% | 6.9% | 86.9% | 1,378.96 | 94.0 | 79.1% |
| shading_lgbm_discrete_hazard | 3,046.49 | 7.7% | 7.7% | 87.2% | 1,370.40 | 93.1 | 79.7% |
| oracle | 3,820.86 | 35.1% | 35.1% | 100.0% | 1,244.07 | 73.7 | 100.0% |

## Landscape quality (test)

| Landscape | Log loss | Brier | ECE | ANLP | Fit, s |
| --- | --- | --- | --- | --- | --- |
| km_global | 0.3196 | 0.0975 | 0.0155 | 3.339 | 0.2 |
| km_segmented | 0.2745 | 0.0844 | 0.0156 | 2.830 | 0.5 |
| lgbm_win_classifier | 0.1986 | 0.0593 | 0.0178 | 3.736 | 41.2 |
| lgbm_discrete_hazard | 0.2050 | 0.0606 | 0.0153 | 2.091 | 25.3 |
