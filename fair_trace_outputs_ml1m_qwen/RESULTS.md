# qwen3.5:9b on MovieLens-1M — results

Users completed: **150**  ·  sensitivity rows 1500  ·  similarity rows 3710  ·  consequence rows 3710

Run finished cleanly. Model calls 9680, cache hits 30, retries 0.

## Localisation

| rule | accuracy | correct |
|---|---|---|
| argmax raw direct | 18.0% | 27/150 |
| argmax direct − noise | 32.7% | 49/150 |

Chance with six labels is 16.7%.

Under the noise-adjusted rule: 88 errors name the wrong stage, 13 are missed detections.

## Signal against the noise floor

| rows | direct | noise | direct − noise |
|---|---|---|---|
| stage carrying the planted fault (236) | 0.4496 | 0.2899 | +0.1597 |
| every other stage (1264) | 0.3668 | 0.3698 | -0.0030 |

Planted stages whose direct effect clears their own noise floor: **163 of 236**.

## Group dispersion

- `age`: identity SNSV = SNSR/2 holds on 1 of 10 rows
- `sex`: identity SNSV = SNSR/2 holds on 10 of 10 rows
- `sim_item`: mean SNSR 0.0353, max 0.0649
- `sim_pref`: mean SNSR 0.0298, max 0.1205

## Consequence

```
          natural  direct  inherited   noise  benefit  benefit_neutral  benefit_delta
stage                                                                                
Elicit     0.6087  0.5837     0.5526  0.5368   0.1262           0.1262         0.0000
Retrieve   0.5403  0.4637     0.5308  0.4694   0.0289           0.0314        -0.0025
Rank       0.6148  0.3442     0.6182  0.3132   0.0348           0.0354        -0.0007
Explain    0.5722  0.1780     0.5577  0.1632   0.5157           0.5013         0.0144
Memory     0.6417  0.3297     0.6416  0.3036   0.5224           0.6785        -0.1526
```

Benefit when the prompt states the user's real attribute versus a counterfactual one:

```
          stated attribute is false  stated attribute is true     gap
stage                                                                
Elicit                       0.1240                    0.1295  0.0054
Retrieve                     0.0283                    0.0297  0.0013
Rank                         0.0342                    0.0356  0.0014
Explain                      0.5189                    0.5111 -0.0077
Memory                       0.5217                    0.5235  0.0019
```

## Figures

- `qwen_ml1m_localisation.png`
- `qwen_ml1m_sensitivity_vs_consequence.png`
- `qwen_ml1m_signal_vs_noise.png`
- `qwen_ml1m_snsr.png`
