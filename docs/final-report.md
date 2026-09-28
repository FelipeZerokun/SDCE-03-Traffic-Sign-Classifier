# Final milestone: frozen model and application

## Model selection (2026-09-28, before test evaluation)

Close this iteration using the existing augmentation-v1 epoch-10 checkpoint.
No additional training or tuning is part of this milestone. Selection uses
validation results only: accuracy 91.26% and macro F1 0.880, compared with
90.84% and 0.864 for baseline-v1. Some classes regressed; see
[augmentation analysis](augmentation-v1.md).

Frozen checkpoint: `outputs/runs/augmentation-v1/best.pt`.
SHA-256: `a16411d65fd4f1b45604f594ac5f2b49e98fc5d81e91c28a49073895f32d4400`.

The original course notebook specifies 93% validation accuracy. This iteration
does not meet that threshold and is being closed as a modern engineering rebuild,
by explicit project-owner decision. Our track-separated split differs from the
course split. We do not claim full original-rubric compliance or recreate the
legacy notebook/HTML submission format.

## Implementation and reproduction

The application loads a 43-class RGB CNN with deterministic full-image 32 x 32
bilinear preprocessing and scaling to [0, 1]. Checkpoint loading validates the
architecture, class order, schema, and preprocessing. Unsupported metadata is
rejected. Prediction returns class IDs, English names, and softmax scores.
These scores are not calibrated confidence estimates.

Evaluation saves `report.json` (aggregate and per-class metrics, per-image
predictions, checkpoint/annotation/manifest fingerprints, device, and audit)
and `confusion_matrix.csv` (rows actual, columns predicted, ordered 0 through 42).
It refuses to overwrite an existing output directory. Test evaluation audits
readability, dimensions, class coverage, repeated paths, and exact decoded RGB
overlap with all training-source images before inference. Near duplicates are
not assessed. ROI validity is outside this full-image preprocessing audit.

From the repository root, after `uv sync --locked`:

The commands below document the original output locations. Evaluation directories
already exist on the completed local project; choose new `--output` paths when
repeating evaluation. The demonstration also requires the ignored local legacy
images and refuses to overwrite its fixed output directory. On a fresh checkout,
restore these local artifacts before reproducing the demonstration.

```powershell
uv run traffic-sign-classifier evaluate --checkpoint outputs/runs/augmentation-v1/best.pt --config configs/dataset.toml --manifest outputs/splits/baseline-v2.json --split validation --output outputs/final/validation
uv run traffic-sign-classifier evaluate --checkpoint outputs/runs/augmentation-v1/best.pt --config configs/dataset.toml --manifest outputs/splits/baseline-v2.json --split test --output outputs/final/test
uv run traffic-sign-classifier predict --checkpoint outputs/runs/augmentation-v1/best.pt --image legacy/Traffic_Sign_Classifier_Project/test_images/test1.png --top-k 5
uv run python scripts/final_demo.py
uv run python scripts/summarize_final.py
```

Inference defaults to CPU; `--device cuda` enables available NVIDIA hardware.
CLI paths resolve relative to the current directory; paths inside dataset TOML
resolve according to its existing configuration rules.
Evaluation and demonstration directories already exist on this machine; choose
new output directories to repeat evaluation. The fixed demonstration script
refuses to overwrite `outputs/final/demo-reviewed`. The summary script rebuilds
the Markdown snapshot and plot from saved results without rerunning inference.

## Final results

The frozen checkpoint reproduced validation accuracy 91.26% and macro F1 0.8802.
CPU test evaluation on 2026-09-28 produced:

| Evaluation | Images | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Full supplied test set | 12,630 | 90.19% | 0.8650 |
| Excluding exact training/validation overlaps | 12,622 | 90.18% | 0.8649 |

All test images decoded as RGB and matched annotated dimensions; all 43 classes
were present. There were no repeated test paths or within-test exact duplicate
groups. Eight test images exactly match training-source images, all from class-14
track 00023. The first integrity gate stopped before inference on discovering
these matches. The evaluator was then extended to enumerate overlaps and report
both the complete supplied set and the subset excluding exact matches, without
changing the model or training data. The subset is not guaranteed free of related
frames or near duplicates; its independence is not established.

The lowest full-set recalls are ice/snow (41.3%), pedestrians (45.0%), and speed
limit 20 km/h (53.3%). Overall accuracy masks these weaknesses. Pedestrian recall
on validation was 71.7%; that gain did not carry over at the same level to test.
No tuning followed these findings.

See [the complete metrics snapshot](final-results.md) for all per-class results.
Local machine-readable reports and the confusion plot are in
`outputs/final/test/`. Rows of the plot are normalized by actual-class support.

## External-image demonstration

The frozen model correctly classified 3 of the 5 historical photos (60%).
Visual review identified an error in the old writeup: image 4 depicts Priority
road, not Yield. The corrected labels are No entry, Road work, Speed limit
(50km/h), Priority road, and Stop. The first, second, and fourth were correct.
The 50 km/h image was predicted as 30 km/h (75.0% softmax score), with the correct
class second (20.8%). Stop was predicted as Yield (46.0%); Stop was absent from
the top five. The source photos retain their backgrounds and were not cropped
again. The small Stop sign occupies only part of its image, a mismatch with the
classifier's intended cropped-sign inputs. This observation does not establish
the cause of the error.

Original URLs and licenses were not recorded. These watermarked historical
photos remain local and are not redistributed. They form a convenience sample,
not a new independent benchmark. Corrected predictions, all top-five scores,
and a reviewed image grid are in `outputs/final/demo-reviewed/`. The initial
`outputs/final/demo/` retains the incorrect legacy label and is superseded.
The [metrics snapshot](final-results.md) records the corrected top-five scores.

## Handoff and limitations

Datasets, checkpoints, and generated reports remain outside Git. Preserve the
local data, split manifest, augmentation-v1 run directory, and final reports
before moving machines. Training reproduction uses `configs/augmentation.toml`
with a new output directory, because existing runs are protected.

Single-seed results and limited validation tracks constrain conclusions.
This classifier expects cropped signs, does not detect signs in road scenes,
and has no unknown-class rejection. Further training, error analysis, calibration,
and more diverse external-image evaluation are deferred to a future iteration.
After reading final test results, future tuning must use validation data and
acknowledge that this test set is no longer an untouched independent holdout.

## Verification and closure

- 116 pytest tests passed, including checkpoint compatibility, softmax/class
  mapping, exact-overlap reporting, source-fingerprint rejection, output
  preservation, and end-to-end evaluation without loading test data.
- Ruff formatting/linting and strict source mypy passed.
- Source distribution and wheel built successfully.
- Real validation evaluation reproduced the existing result; the prediction CLI
  smoke test succeeded. Frozen checkpoint fingerprint remained unchanged.
- The existing GitHub Actions workflow covers these quality checks and packaging.
  Remote run status is recorded in the repository's
  [Actions history](https://github.com/FelipeZerokun/SDCE-03-Traffic-Sign-Classifier/actions/workflows/ci.yml).

The local engineering milestone is complete. The final milestone is delivered
on the `modernize-python` branch; its remote verification is available in the
Actions history above. Further model improvement is explicitly deferred.
