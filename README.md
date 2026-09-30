# MAIR Part 1a – Dialog act classification

## Project layout

```
data/dailog_acts.dat            the original dataset
data/difficult_disfluent.dat    a small dataset of our hard to classify original utterances (stuttering, wrong spelling etc.)
data/difficult_negation.dat     a small dataset of our hard to classify original utterances (negation)
dialog_acts.py                  loading the .dat file, original and grouped 85/15 splits
classifiers.py                  rule based baseline, SVM, MLP, decision tree + loading saved models
embeddings.py                   frozen DistilBERT sentence embeddings
train.py                        trains and evaluates all models, saves models/ and results/
evaluate.py                     evaluates the trained models on a held-out .dat file
analysis.py                     evaluation for the report (error analysis, difficult cases, leakage)
chat.py                         interactive prompt to classify your own utterances
requirements.txt                python dependencies
models/                         saved models created by train.py
results/                        metrics, predictions and plot created by train and evaluate.py
```

## 1. Install

With Python 3.11, from the project folder:

```
pip install -r requirements.txt
```


## 2. Train (reproduce our results)

We have left our pre-trained models under the `models/` folder, in case the training wants to be skipped and the evaluation/chat just run directly after unzipping. If that is not the case however, then the below steps are how to build the models from scratch (and you can delete the models included upon extracting the zip file).

Run from the project folder:

```
python train.py
```

This trains every model on both splits, prints accuracy, balanced accuracy and a per-class
report for each, and saves:

- `models/` – the models used by `evaluate.py` and `chat.py` 
- `results/all_results.csv` – accuracy and balanced accuracy of every model/feature/split
- `results/svm_results.csv`, `mlp_results.csv`, `dt_results.csv` – per classifier results
- `results/model_comparison.png` – bar plot comparing all models

Computing the DistilBERT embeddings takes about 20 minutes on a laptop CPU (much less on a GPU).
For a quick run (under a minute) with only the rule based and BoW models, use:

```
python train.py --skip-embeddings
```

All randomness uses a fixed seed (`SEED = 12345` in `dialog_acts.py`), so results are the same on every run.

## 3. Evaluate on the held-out test set

Train first (step 2), then:

```
python evaluate.py --data path/to/dialog_acts_test.dat
```

The file must have the same format as `dialog_acts.dat`.
Every trained model is evaluated (rule, svm, mlp, dt, and svm_emb, mlp_emb, dt_emb if the
embedding models were trained). To evaluate one model only:

```
python evaluate.py --data path/to/dialog_acts_test.dat --model svm
```

For each model it prints accuracy, balanced accuracy, macro F1 and a per-class report,
followed by a summary table. The results are saved to:

- `results/heldout_results.csv` – accuracy, balanced accuracy and macro F1 per model

Empty lines and lines without an utterance are skipped with a warning.

## 4. Evaluation for the report

```
python analysis.py
```

Trains every model on both splits again and saves to `results/`:

- `eval_metrics.csv` – accuracy, balanced accuracy, macro and weighted F1 (plus a majority class baseline)
- `eval_per_class_f1.csv`, `eval_confusions.csv` – F1 per dialog act and the most frequent confusions (grouped split)
- `eval_hard_utterances.csv` – test utterances that every system gets wrong
- `eval_difficult.csv`, `eval_difficult_<set>.csv` – results on our own difficult cases in `data/difficult_negation.dat` and `data/difficult_disfluent.dat`
- `eval_leakage.csv` – accuracy on test sentences that were / weren't seen during training

The DistilBERT embeddings of the unique utterances are cached in `models/embedding_cache.joblib`
(first run about 3 minutes on a CPU). Use `--skip-embeddings` for the rule based and BoW models only.

## 5. Interactive prompt

```
python chat.py
```

Type a sentence to see its predicted dialog act. Input is converted to lower case and
punctuation is removed before classifying. Commands:

| Command         | What it does                                 |
|-----------------|----------------------------------------------|
| `:model <name>` | switch classifier: `rule`, `svm`, `mlp`, `dt` |
| `:models`       | show the available classifiers               |
| `:help`         | show the commands                            |
| `quit` / `exit` | stop (Ctrl+C also works)                     |

Start with a different classifier with `python chat.py --model mlp`. Example:

```
Using classifier: svm
> I want cheap food in the north!
dialog act: inform  (classifier: svm)
> :model rule
Switched to classifier: rule
> thank you, goodbye
dialog act: thankyou  (classifier: rule)
> zoot doot
dialog act: null  (classifier: rule)
> quit
Goodbye!
```

If none of the words of an input were seen during training, the BoW classifiers still give a
prediction but add a note that it is unreliable.

## Notes

- All data and user input is lower cased.
- Out-of-vocabulary words are dropped by the BoW vectorizer (the vocabulary is learned on the train set only).