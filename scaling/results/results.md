Held-out mean return (mean of the last 3 checkpoints; min–max in brackets) and failure rate.

| Run | Data (h) | Unseen motions | Unseen performers | LAFAN1 | Own training motions |
|---|---:|---:|---:|---:|---:|
| D1 | 1.0 | 0.867 [0.865–0.870], fail 1.8% | 0.872 [0.871–0.873], fail 3.8% | 0.624 [0.616–0.630], fail 15.9% | 0.960 [0.958–0.961], fail 0.0% |
| D7 | 7.0 | 0.909 [0.904–0.916], fail 0.8% | 0.918 [0.912–0.923], fail 0.6% | 0.706 [0.699–0.719], fail 14.1% | 0.940 [0.932–0.947], fail 0.0% |
| D40 | 40.0 | 0.926 [0.922–0.929], fail 0.5% | 0.929 [0.925–0.932], fail 0.1% | 0.755 [0.725–0.771], fail 11.5% | 0.942 [0.937–0.945], fail 0.0% |
| Dall | 254.4 | 0.926 [0.923–0.929], fail 0.5% | 0.930 [0.925–0.934], fail 0.1% | 0.779 [0.778–0.779], fail 7.4% | 0.919 [0.913–0.924], fail 0.1% |

Best single checkpoint (chosen by looking at the same evaluation, so optimistic) and its iteration.

| Run | Unseen motions | Unseen performers | LAFAN1 | Own training motions |
|---|---:|---:|---:|---:|
| D1 | 0.870 @ 40k | 0.873 @ 36k | 0.630 @ 40k | 0.961 @ 36k |
| D7 | 0.916 @ 40k | 0.923 @ 40k | 0.719 @ 40k | 0.947 @ 40k |
| D40 | 0.929 @ 40k | 0.932 @ 40k | 0.771 @ 36k | 0.945 @ 40k |
| Dall | 0.929 @ 36k | 0.934 @ 36k | 0.779 @ 36k | 0.924 @ 36k |

World-model 8-step error (normalised, x1e-3): pooled held-out windows vs the run's own training rollouts.

| Run | Pooled held-out | Own training rollouts (online) | it/s | GPU-h trained |
|---|---:|---:|---:|---:|
| D1 | 44.5 | 7.2 | 2.33 | 4.89 |
| D7 | 21.3 | 9.8 | 2.30 | 5.02 |
| D40 | 17.9 | 10.7 | 2.30 | 5.19 |
| Dall | 17.7 | 13.0 | 2.33 | 4.98 |

Compute to a mean return of 0.7 (GPU-hours; iterations in brackets; – = not reached).

| Run | Unseen motions | Unseen performers | LAFAN1 | Own training motions |
|---|---:|---:|---:|---:|
| D1 | 0.64 (4000) | 0.64 (4000) | – | 0.64 (4000) |
| D7 | 0.65 (4000) | 0.65 (4000) | 3.26 (25597) | 0.65 (4000) |
| D40 | 0.67 (4000) | 0.67 (4000) | 2.30 (16371) | 0.67 (4000) |
| Dall | 0.65 (4000) | 0.65 (4000) | 1.59 (11906) | 0.65 (4000) |

Compute to a mean return of 0.8 (GPU-hours; iterations in brackets; – = not reached).

| Run | Unseen motions | Unseen performers | LAFAN1 | Own training motions |
|---|---:|---:|---:|---:|
| D1 | 0.85 (5806) | 0.88 (5976) | – | 0.64 (4000) |
| D7 | 0.91 (6158) | 0.88 (5875) | – | 0.85 (5642) |
| D40 | 0.95 (4947) | 0.91 (4651) | – | 0.85 (4102) |
| Dall | 0.80 (5245) | 0.79 (5213) | – | 0.96 (6610) |

Compute to a mean return of 0.85 (GPU-hours; iterations in brackets; – = not reached).

| Run | Unseen motions | Unseen performers | LAFAN1 | Own training motions |
|---|---:|---:|---:|---:|
| D1 | 1.39 (10357) | 1.41 (10536) | – | 0.84 (5684) |
| D7 | 1.10 (7772) | 1.11 (7820) | – | 1.06 (7390) |
| D40 | 1.25 (7428) | 1.21 (7137) | – | 1.15 (6658) |
| Dall | 1.03 (7254) | 1.03 (7204) | – | 1.28 (9287) |

Compute to a mean return of 0.9 (GPU-hours; iterations in brackets; – = not reached).

| Run | Unseen motions | Unseen performers | LAFAN1 | Own training motions |
|---|---:|---:|---:|---:|
| D1 | – | – | – | 1.09 (7800) |
| D7 | 2.40 (18602) | 2.19 (16787) | – | 1.58 (11685) |
| D40 | 2.31 (16384) | 2.12 (14859) | – | 1.69 (11190) |
| Dall | 2.25 (17419) | 2.05 (15752) | – | 2.66 (20823) |
