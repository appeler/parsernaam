Accuracy on as-is input (NC is all caps):

| source | task | majority | frequency | published | fixed_inference | retrained |
| --- | --- | --- | --- | --- | --- | --- |
| FL | ordering | 50.0% | 46.2% | 85.1% | 98.5% | 92.7% |
| FL | single_distinct_name | 49.8% | 45.2% | 79.4% | 86.5% | 81.6% |
| FL | single_record_weighted | 51.0% | 49.2% | 73.6% | 81.8% | 84.4% |
| NC | ordering | 50.0% | 87.1% | 53.2% | 98.4% | 96.2% |
| NC | single_distinct_name | 41.4% | 66.9% | 51.2% | 87.3% | 79.6% |
| NC | single_record_weighted | 50.0% | 84.9% | 54.2% | 80.9% | 90.9% |

Expected calibration error of raw scores (lower is better):

| source | task | majority | frequency | published | fixed_inference | retrained |
| --- | --- | --- | --- | --- | --- | --- |
| FL | ordering | 0.000 | 0.433 | 0.130 | 0.008 | 0.020 |
| FL | single_distinct_name | 0.013 | 0.303 | 0.069 | 0.056 | 0.063 |
| FL | single_record_weighted | 0.001 | 0.399 | 0.096 | 0.129 | 0.060 |
| NC | ordering | 0.000 | 0.100 | 0.430 | 0.007 | 0.003 |
| NC | single_distinct_name | 0.097 | 0.066 | 0.224 | 0.087 | 0.084 |
| NC | single_record_weighted | 0.010 | 0.087 | 0.151 | 0.142 | 0.008 |

Release gates: case_invariance=True, case_violations=0, retrained_beats_both_baselines=True, fixed_beats_published=True.
