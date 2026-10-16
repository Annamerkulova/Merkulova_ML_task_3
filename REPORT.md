# Model card: who should the bank's call centre phone?

## 1. Purpose

A bank phones clients to sell a term deposit. The call centre can phone only **10% of the clients**. The model gives every client a score (how likely they are to say "yes"). We sort clients by this score and phone the top 10%.

The model is for this one use: ranking clients of this bank's phone campaign. It should not be used to decide anything about a single person (for example credit), or for other banks or products.

## 2. Data

- `train.csv`: 32,940 clients, 22 columns. `test.csv`: 8,236 clients, 20 columns.
- Target `y`: 3,711 clients said "yes" (11.27%) and 29,229 said "no". A model that always says "no" is right 88.7% of the time but finds no one, so accuracy is not a good metric here.
- `duration` (how long the call lasted) is only in `train.csv`. We only know it **after** the call, so we do not use it. If we used it, the model would "see the answer" (data leakage).
- Some values really mean "missing": the word `unknown` in six columns (`default` 6,871 times, `education` 1,401, `housing` 794, `loan` 794, `job` 275, `marital` 64), and `pdays = 999` in 31,694 rows (96.2%), which means "not contacted before".
- [TODO: one sentence about what you found with `pd.crosstab(train["pdays"] == 999, train["previous"] > 0)`.]

## 3. How the model is built

- I put 20% of `train.csv` aside as my own **hold-out** (6,588 clients) and used it only once, at the end. All comparing and tuning used 5-fold cross-validation on the other 80%. Folds are stratified, the seed is 2305.
- All preparation steps are inside one scikit-learn `Pipeline`, so they learn only from the training part of each fold. (I checked: in every fold the scaler's mean equals the mean of that fold's training rows.)
- Numbers are scaled. `pdays = 999` is replaced by 0 and a new yes/no column says "not contacted before". Text columns are one-hot encoded; `unknown` is kept as its own category.
- I compared these choices with cross-validation (`pdays` as a plain number or with the flag; `unknown` as a category or filled in). The differences were far smaller than the noise between folds (for example 0.0007 in average precision, while the fold noise is about 0.02), so none of them is clearly better. I kept the flag and the `unknown` category because they are the simplest to explain.
- Models tried: logistic regression, decision tree, random forest and histogram gradient boosting. Settings were tuned by cross-validation on the training part only. 
- Final model: gradient boosting (`learning_rate` 0.03, `max_leaf_nodes` 15, `min_samples_leaf` 20), trained again on all of `train.csv`.

## 4. Metrics and results

**Which metric.** Because we phone only 10%, what matters is whether the **top of the list** contains many "yes" clients. So I use average precision (to tune), ROC-AUC (to compare models), and precision, recall and lift for the top 10%. Lift means: how many times better than phoning clients at random.

**Cross-validation (mean ± spread over 5 folds).The simple baselines:

| Model | accuracy | ROC-AUC | avg. precision | precision top 10% |
|---|---|---|---|---|
| always "no" | 0.8873 | 0.50 | 0.113 | 0.112 (lift 0.99) |
| logistic regression | 0.9007 | 0.788 ± 0.013 | 0.451 ± 0.024 | 0.501 ± 0.024 (lift 4.44) |

Accuracy hardly changes (0.887 to 0.901), but the model is about 4 times better at finding "yes" clients.

**Which model is better?** Boosting was better than logistic regression in all 5 folds for average precision (+0.016) and ROC-AUC (+0.011). This is real but small. For precision in the top 10% (the number the call centre cares about) the difference (+0.008) is smaller than the noise, so we cannot say which model is better for this. Boosting against random forest: average precision is inside the noise; ROC-AUC is +0.006 in all folds, which is tiny. I chose boosting because it had the best average precision and a steady lead in ROC-AUC, but I do not claim it is clearly the best for the 10% rule.

**Class imbalance.** I tried `class_weight="balanced"`. It did not help: for logistic regression the average precision was a bit worse (−0.004), for the other models the changes were inside the noise or tiny. I did not use it.

**Hold-out (used once).** ROC-AUC 0.82 and average precision 0.50 (random guessing: 0.11).

**The 10% rule on the hold-out.** We phone 659 of 6,588 clients.

| | not phoned | phoned |
|---|---|---|
| really "no" | 5,541 | 305 |
| really "yes" | 388 | 354 |

For the call-centre manager: of 659 calls, 354 end in a "yes" (54%, with a standard error of about 2 points). 305 calls bring nothing. Phoning 659 random clients would give only about 74 "yes", so the model finds 4.8 times more (lift 4.77). We reach 48% of all interested clients (354 of 742). The other 388 are not phoned. That is the price of a 10% limit: even a perfect model could reach at most about 89%.

**Predictions.** In `predictions.csv`, the 824 test clients (10%) with the highest scores have `y_pred = 1`. I use the rank, not a fixed score, because the real limit is the number of calls.

## 5. Limits and ways the model can fail

1. **Time is hidden in the data.** The economic columns (`euribor3m`, `nr.employed`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`) are almost the same for all clients phoned in the same period, so they work like a "period number". For example, in the lowest fifth of `euribor3m` 29.9% said "yes", against 11.3% overall. If rates and client behaviour change, the model can get worse without any warning.
2. **My split flatters the result.** The data were split at random, so clients from the same period are in both training and hold-out. In real life we train on the past and phone in the future, where the period is new. The numbers above are probably too good. 
3. **The same person may be in training and hold-out.** There is no client ID, so I cannot check it.
4. **Small hold-out.** The precision for the top 10% is measured on 659 clients (standard error about 2 points).
5. **Limited search.** Several best settings were at the edge of the search grid, so better settings may exist (the differences between top models are small anyway).
6. **Missing values.** `default = unknown` appears for 21% of clients. It may not be random, and the model may use it in a way I did not check.

## 6. What to watch after launch

- The share of "yes" among phoned clients. I expect about 50–55%. If it falls a lot, something changed.
- The distribution of the economic columns and of `unknown`/`pdays` values against the training data.
- Each month, compare the model's scores with real results and retrain when it gets worse. Check the model on new periods, not only on random parts of old data.