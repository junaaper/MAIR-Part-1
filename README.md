# MAIR Part 1 – Dialog act classification (1a) and restaurant recommendation dialog system (1b)

## Project layout

```
data/dailog_acts.dat                the original dataset
data/difficult_disfluent.dat        a small dataset of our hard to classify original utterances (stuttering, wrong spelling etc.)
data/difficult_negation.dat         a small dataset of our hard to classify original utterances (negation)
data/restaurant_info.csv            restaurant database (Part 1b)
data/restaurant_info_extended.csv   restaurant database with food quality, crowdedness and length of stay (reasoning)
data/matcher_tests.txt              hand-labelled typos and paraphrases for comparing the slot matchers
reference_dialogs.txt               the 20 reference dialogs given with the assignment

dialog_acts.py                      loading the .dat file, original and grouped 85/15 splits
classifiers.py                      rule based baseline, SVM, MLP, decision tree + loading saved models
embeddings.py                       frozen DistilBERT sentence embeddings
train.py                            trains and evaluates all models, saves models/ and results/
evaluate.py                         evaluates the trained models on a held-out .dat file
analysis.py                         evaluation for the report (error analysis, difficult cases, leakage)
chat.py                             interactive prompt to classify your own utterances

dialog_manager.py                   the dialog system: state transition function, templates, lookup, logging (Part 1b)
slot_extraction.py                  preferences and requests from an utterance: keywords + Levenshtein / semantic fallback
reasoning.py                        the six inference rules and contradiction handling
compare_matchers.py                 compares the Levenshtein and semantic slot matchers
run_reference_dialogs.py            replays the reference dialogs through the dialog system

logs/example_dialog.jsonl           example interaction log
requirements.txt                    python dependencies
models/                             saved models created by train.py
results/                            metrics, predictions and plots created by the scripts above
```

## 1. Install

With Python 3.11, from the project folder:

```
pip install -r requirements.txt
```

The dialog system only needs torch and DistilBERT when it is started with `--matcher semantic`.


# Part 1a – Dialog act classification

## 2. Train (reproduce our results)

We have left our pre-trained models under the `models/` folder, in case the training wants to be skipped and the evaluation/chat just run directly after unzipping. If that is not the case however, then the below steps are how to build the models from scratch (and you can delete the models included upon extracting the zip file).

Run from the project folder:

```
python train.py
```

This trains every model on both splits, prints accuracy, balanced accuracy and a per-class
report for each, and saves:

- `models/` – the models used by `evaluate.py`, `chat.py` and `dialog_manager.py`
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

The file can be anywhere on disk and must have the same format as `data/dailog_acts.dat`
(one `dialog_act utterance` per line); several files can be given at once.
Every trained model is evaluated (rule, svm, mlp, dt, and svm_emb, mlp_emb, dt_emb if the
embedding models were trained). To evaluate one model only:

```
python evaluate.py --data path/to/dialog_acts_test.dat --model svm
```

For each model it prints accuracy, balanced accuracy, macro F1 and a per-class report,
followed by a summary table. The results are saved to:

- `results/heldout_results.csv` – accuracy, balanced accuracy and macro F1 per file and model

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


# Part 1b – Restaurant recommendation dialog system

## 6. Run the dialog system

```
python dialog_manager.py
```

This uses the SVM from Part 1a (`models/svm.joblib`, included). Type your answers after `user:`;
the dialog ends when you say goodbye (Ctrl+C also stops it). Options:

| Option                                  | What it does                                                                 |
|-----------------------------------------|------------------------------------------------------------------------------|
| `--matcher levenshtein\|semantic`       | fallback for slot extraction when keyword matching fails (default: levenshtein) |
| `--transparency on\|off`                | show the reasoning behind a recommendation, or only the restaurant (default: on) |
| `--recommendation-mode patient\|eager`  | wait for all preferences before recommending, or recommend as soon as one is known (default: patient) |
| `--restaurants CSV`                     | use another restaurant database (default: `data/restaurant_info_extended.csv`) |
| `--seed N`                              | random seed for choosing between matching restaurants                      |
| `--debug`                               | print the dialog act, state and preferences every turn                     |
| `--no-log`                              | do not write an interaction log                                              |
| `--log-dir DIR`                         | where to write the interaction logs (default: `logs/`)                       |

`--transparency` and `--recommendation-mode` are the configurable features.
A file given with `--restaurants` needs the columns `restaurantname`, `pricerange`, `area` and `food`;
the additional requirements also need `food quality`, `crowdedness` and `length of stay`.


## 7. Reproduce the evaluation

Compare the two slot extraction fallbacks on `data/matcher_tests.txt`:

```
python compare_matchers.py            # add --sweep to also try semantic thresholds
```

This saves `results/matcher_comparison.csv` (accuracy, false matches and speed per matcher) and
`results/matcher_predictions.csv` (every prediction).

Replay the 20 reference dialogs through the system:

```
python run_reference_dialogs.py
python run_reference_dialogs.py --recommendation-mode eager
python run_reference_dialogs.py --file path/to/dialogs.txt --restaurants path/to/restaurants.csv
```

It prints the expected and our system utterances side by side, with the dialog act and state of every
turn. `--file` takes any dialog file in the format of `reference_dialogs.txt`.

## 8. Interaction logs

Every run writes `logs/dialog_<date>_<time>.jsonl`, one JSON record per line (not part of the repository);
`logs/example_dialog.jsonl` is an example. The records are:

| Event    | Fields |
|----------|--------|
| `config` | `classifier`, `matcher`, `transparency`, `seed`, `reasoning` (reasoning columns found), `recommendation_mode`, `restaurants` |
| `system` | `turn` (0), `state`, `utterance` – the welcome message |
| `turn`   | `turn`, `user` (what was typed), `act` (predicted dialog act), `state_before`, `state_after`, `prefs` (area, food, pricerange), `extra` (additional requirements), `suggested` (restaurant suggested so far), `system` (the reply) |
| `end`    | `turns`, `state` (6 if the user said goodbye), `prefs`, `extra`, `suggested` |


## Notes

- All data and user input is lower cased.
- Out-of-vocabulary words are dropped by the BoW vectorizer (the vocabulary is learned on the train set only).
- The dialog system writes a log for every run unless `--no-log` is given.