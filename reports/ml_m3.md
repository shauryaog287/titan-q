# M3: NIFTY open-to-close direction from pre-open information

Implements reports/preregistration_ml_models.md (M3). Walk-forward, refit every 21 days; long if P(up) > 0.55, short if < 0.45, 2 bps per side.

Features: 25; first prediction 2016-11-25.

## Development

| Model | Days | Traded | Accuracy (traded) | Accuracy (all days) | Mean net/day | t | Sharpe |
|---|---|---|---|---|---|---|---|
| p_logit | 1010 | 644 | 52.2% | 51.9% | +0.6 bps | +0.25 | 0.12 |
| p_gbm | 1010 | 746 | 52.4% | 51.8% | +1.0 bps | +0.38 | 0.19 |
| always long | 1010 | 1010 | - | 45.7% (up days) | -12.6 bps | - | -2.09 |

**Decision: no model qualifies; test stays sealed.**

