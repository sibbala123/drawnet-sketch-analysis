This is a fantastic run! I've analyzed your train_log.csv and test_metrics.csv files. Here is the breakdown of how your model performed, with some very interesting insights:

1. Verification of Classification Accuracies (Test Set)
Your final evaluation on the unseen test set yielded incredibly strong results across both tasks.

Intent Recognition (30 Categories):

Top-1 Accuracy: 88.6% — Getting almost 9 out of 10 sketches exactly right across 30 distinct categories is very impressive.
Top-5 Accuracy: 98.0% — When the model is wrong, the correct answer is almost always in its top 5 guesses!
Macro F1: 0.872 — Shows excellent balance, meaning it performs well even on those TU-Berlin classes that had fewer images compared to the massive QuickDraw classes.
Deviation Detection (4 Geometric Irregularities):

Exact Match (Subset) Accuracy: 92.1% — For 92% of the images, the model perfectly identified the exact combination of the 4 deviations present.
Macro F1: 0.901
AUROC Scores:
Rotation: 0.998
Closure Failure: 0.999
Spatial Disorganization: 0.996
Size Distortion: 0.926 (We can see clearly here that size distortion is the hardest deviation for the model to detect. This makes sense instinctively, as judging "abnormal size" highly depends on the varying original canvas sizes between QuickDraw and TU-Berlin).
2. Is there Overfitting?
Yes, there is some mild overfitting in the later epochs.

If you look closely at the trajectory in your train_log.csv:

Epoch 1–30: Both training loss and validation loss decrease beautifully in tandem. By epoch 30, validation loss is around 0.508.
Epoch 30–50: The training loss continues to plunge rapidly (going from 0.105 down to 0.062), which means the model is getting extremely confident on the training data.
However, your validation loss stops dropping and actually begins creeping up slightly (bouncing between 0.510 and 0.530).
The Good News: Even though the loss creeps up (meaning the model becomes more stubbornly confident about some wrong test predictions), the actual validation accuracy doesn't decay. It stays permanently hovering around ~89.4%.

Since your script saves the best.pt automatically based on the lowest validation loss, the checkpoint you saved to your Drive probably snagged the weights from securely around Epoch 25 to 30, safely escaping the overfitting of the final 20 epochs!

Overall Verdict
The new pipeline redesign works perfectly. The 88.6% intent accuracy paired with >90% deviation identification proves that your ResNet-50 shared backbone is highly capable of tackling multi-task learning without the heads fighting for resources.

