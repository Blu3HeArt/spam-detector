# SMS Spam Detector

A beginner machine-learning project that classifies SMS messages as
spam or legitimate messages (ham), then tests how wording changes
affect its predictions.

## Tools

- Python 3.12.3
- scikit-learn 1.9.1
- TF-IDF for converting text into numerical features
- Multinomial Naive Bayes for classification

## Dataset

Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Credit: Almeida, T. & Hidalgo, J. (2011).
SMS Spam Collection. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CC84

License: Creative Commons Attribution 4.0 (CC BY 4.0).

- Original messages loaded: 5,574
- Exact duplicate messages removed: 403
- Unique messages: 5,171
- Training messages: 4,136
- Test messages: 1,035

Duplicates were removed before splitting the data. The split used
80% for training and 20% for testing, with stratification and
random_state=42. The TF-IDF vocabulary was learned only from
training data.

## Test Results

| Metric | Result |
|---|---:|
| Accuracy | 95.3% |
| Spam precision | 100% |
| Spam recall | 62.6% |
| Spam F1-score | 0.770 |
| Macro F1-score | 0.872 |

| Outcome | Messages |
|---|---:|
| Legitimate messages correctly classified | 904 |
| Legitimate messages incorrectly flagged as spam | 0 |
| Spam messages correctly detected | 82 |
| Spam messages missed | 49 |

Always predicting ham would achieve 87.3% accuracy on this test set.
The model improves on that baseline, but still misses 37.4% of spam.
These results describe this test set, not guaranteed future performance.

## Wording Experiment

I tested three versions of a fictional, unsolicited prize offer:

| Message | Prediction |
|---|---|
| Congratulations! You won a FREE cash prize. Call now to claim! | spam |
| Congratulations! You won a FR33 c@sh pr1ze. Call now to claim! | spam |
| Hey, you were selected to receive some money. Give us a ring to collect it. | ham |

The spelling changes did not bypass detection, but the conversational
rewrite did. This demonstrates one example of evasion, not a universal
bypass. The model sees message text and cannot verify the sender or offer.

## How to Run

1. Create and activate a Python virtual environment.
2. Install the library:

   ```bash
   python -m pip install scikit-learn==1.9.1
   ```

3. Download and extract the UCI dataset so the message file is at:

   ```text
   data/SMSSpamCollection
   ```

4. Run:

   ```bash
   python inspect_data.py
   ```

The script loads the data, removes duplicates, trains the model,
prints evaluation results, and runs the example-message experiments.
It retrains each time it runs; it does not save a trained model.

## What I Learned

- How labeled examples support supervised learning.
- How TF-IDF converts text into numerical features.
- Why training and test data must stay separate.
- How duplicates can cause data leakage.
- Why accuracy alone can hide weak spam detection.
- How precision and recall describe different mistakes.
- How rewording a message can change a model's prediction.

## Limitations

This is an educational baseline, not a production spam filter.
It uses an older SMS dataset, misses some spam, and was tested with
only a few handcrafted messages. Performance on modern messages,
different populations, or other languages has not been established.