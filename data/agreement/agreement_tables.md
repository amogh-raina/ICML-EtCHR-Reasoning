## Human inter-annotator agreement (3 annotators)

| Criterion | Krippendorff α | Exact % | Within-1 % | n pairs |
|---|---|---|---|---|
| Step occurrence (nominal) | 0.139 | 96.9 | 100.0 | 720 |
| Step comprehensiveness (ordinal, occurred-only) | 0.085 | 41.4 | 87.4 | 701 |
| Overall conciseness (ordinal) | 0.106 | 36.7 | 90.6 | 180 |

## Judge ↔ human-consensus alignment

| Judge | Compr. ρ | Compr. bias (J−H) | Compr. within-1 | Concise ρ | Concise bias | Occurrence %agree |
|---|---|---|---|---|---|---|
| claude | 0.156 | -0.26 | 85.8 | 0.235 | -0.54 | 97.3 |
| deepseek | 0.239 | -0.05 | 81.9 | 0.285 | +0.44 | 96.4 |
| gpt5.5 | 0.326 | -0.01 | 85.4 | 0.229 | +0.01 | 97.8 |

Judge–judge α: comprehensiveness=0.406, conciseness=0.076

(ρ = Spearman across matched items; bias = mean(judge − human consensus), positive = judge more lenient.)
