# Final results snapshot

Checkpoint SHA-256: `a16411d65fd4f1b45604f594ac5f2b49e98fc5d81e91c28a49073895f32d4400`.

| Evaluation | Images | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Full supplied test set | 12630 | 90.19% | 0.8650 |
| Excluding exact training/validation overlaps | 12622 | 90.18% | 0.8649 |

The audit found 8 overlapping test images in 8 exact RGB groups. Within-test duplicate groups: 0. Near-duplicate independence was not assessed.

## Full supplied test set: per-class results

| Class | Name | Support | Precision | Recall | F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Speed limit (20km/h) | 60 | 0.6809 | 0.5333 | 0.5981 |
| 1 | Speed limit (30km/h) | 720 | 0.8413 | 0.9500 | 0.8924 |
| 2 | Speed limit (50km/h) | 750 | 0.9700 | 0.9040 | 0.9358 |
| 3 | Speed limit (60km/h) | 450 | 0.8320 | 0.8911 | 0.8605 |
| 4 | Speed limit (70km/h) | 660 | 0.8826 | 0.9000 | 0.8912 |
| 5 | Speed limit (80km/h) | 630 | 0.7900 | 0.9254 | 0.8523 |
| 6 | End of speed limit (80km/h) | 150 | 0.9483 | 0.7333 | 0.8271 |
| 7 | Speed limit (100km/h) | 450 | 0.8416 | 0.8622 | 0.8518 |
| 8 | Speed limit (120km/h) | 450 | 0.8960 | 0.8044 | 0.8478 |
| 9 | No passing | 480 | 0.9781 | 0.9292 | 0.9530 |
| 10 | No passing for vehicles over 3.5 metric tons | 660 | 0.9772 | 0.9758 | 0.9765 |
| 11 | Right-of-way at the next intersection | 420 | 0.8698 | 0.8905 | 0.8800 |
| 12 | Priority road | 690 | 0.9840 | 0.9826 | 0.9833 |
| 13 | Yield | 720 | 0.9521 | 0.9931 | 0.9721 |
| 14 | Stop | 270 | 0.9669 | 0.9741 | 0.9705 |
| 15 | No vehicles | 210 | 0.9498 | 0.9905 | 0.9697 |
| 16 | Vehicles over 3.5 metric tons prohibited | 150 | 0.9762 | 0.8200 | 0.8913 |
| 17 | No entry | 360 | 1.0000 | 0.7778 | 0.8750 |
| 18 | General caution | 390 | 0.9528 | 0.7769 | 0.8559 |
| 19 | Dangerous curve to the left | 60 | 0.9825 | 0.9333 | 0.9573 |
| 20 | Dangerous curve to the right | 90 | 0.5528 | 0.9889 | 0.7092 |
| 21 | Double curve | 90 | 0.8085 | 0.8444 | 0.8261 |
| 22 | Bumpy road | 120 | 0.8425 | 0.8917 | 0.8664 |
| 23 | Slippery road | 150 | 0.7692 | 0.6667 | 0.7143 |
| 24 | Road narrows on the right | 90 | 0.7821 | 0.6778 | 0.7262 |
| 25 | Road work | 480 | 0.9779 | 0.9229 | 0.9496 |
| 26 | Traffic signals | 180 | 0.7647 | 0.8667 | 0.8125 |
| 27 | Pedestrians | 60 | 0.4500 | 0.4500 | 0.4500 |
| 28 | Children crossing | 150 | 0.7062 | 0.9133 | 0.7965 |
| 29 | Bicycles crossing | 90 | 0.8571 | 0.8000 | 0.8276 |
| 30 | Beware of ice/snow | 150 | 0.6526 | 0.4133 | 0.5061 |
| 31 | Wild animals crossing | 270 | 0.8653 | 0.9519 | 0.9065 |
| 32 | End of all speed and passing limits | 60 | 0.8824 | 1.0000 | 0.9375 |
| 33 | Turn right ahead | 210 | 0.9810 | 0.9857 | 0.9834 |
| 34 | Turn left ahead | 120 | 1.0000 | 0.8667 | 0.9286 |
| 35 | Ahead only | 390 | 0.9622 | 0.9795 | 0.9708 |
| 36 | Go straight or right | 120 | 0.9907 | 0.8917 | 0.9386 |
| 37 | Go straight or left | 60 | 0.9355 | 0.9667 | 0.9508 |
| 38 | Keep right | 690 | 0.9461 | 0.9913 | 0.9682 |
| 39 | Keep left | 90 | 0.9870 | 0.8444 | 0.9102 |
| 40 | Roundabout mandatory | 90 | 0.9639 | 0.8889 | 0.9249 |
| 41 | End of no passing | 60 | 0.9773 | 0.7167 | 0.8269 |
| 42 | End of no passing by vehicles over 3.5 metric tons | 90 | 0.9419 | 0.9000 | 0.9205 |

## Historical external images

Full photos are used without additional cropping. Original source URLs/licenses are unknown; images remain local and are not redistributed.

Image 4 was visually relabeled Priority road (12), correcting the old Yield label.

Top-1 accuracy: 3/5 (60%).

| Image | Actual class ID | Top five class IDs and softmax scores |
| --- | ---: | --- |
| test1.png | 17 | 17: 0.9876; 12: 0.01239; 14: 3.215e-07; 20: 6.243e-10; 32: 3.852e-10 |
| test2.png | 25 | 25: 1; 22: 1.05e-08; 30: 3.847e-09; 28: 3.074e-09; 29: 2.387e-09 |
| test3.png | 2 | 1: 0.75; 2: 0.2078; 0: 0.02253; 5: 0.0195; 16: 0.0002178 |
| test4.png | 12 | 12: 1; 10: 8.031e-13; 42: 4.929e-16; 32: 2.541e-20; 26: 9.183e-22 |
| test5.png | 14 | 13: 0.4602; 20: 0.3621; 17: 0.1721; 10: 0.005416; 12: 9.534e-05 |

Scores are softmax outputs, not calibrated confidence. This small historical sample is not an independent benchmark.
