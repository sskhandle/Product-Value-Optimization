# Product Value Optimization

Agents that decide which products to buy, to make as much money as possible. Built in three phases for a course project.

## How it works

Every product has a value (1000), a price and a hidden label: good (works, pays the value) or bad (pays nothing). An agent's wealth is the value of the good products it bought minus what it paid.

1. **Phase 1:** No data, only a probability that the product works. `Agent_skatkar` buys when `prob > price / value`. Tested in three markets: fair, junk yard and fancy.
2. **Phase 2:** Products have features. `Agent_skatkar` fits about 40 scikit-learn classifiers (kNN, trees, forests, SVC, discriminant analysis, etc.) and keeps the one with the best validation accuracy. The simulator then buys at several prices using `value * prob > price`.
3. **Phase 3:** The agent starts with one good and one bad product and picks products one at a time, up to half of the list. Before each pick it refits `BernoulliNB` on what it owns, then takes the product with the highest `prob * (value - price)`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy scikit-learn
```

The code is Python 3. Run each script from inside its phase folder, since datasets are read from `./`. Copy the dataset files you need from that phase's `Datasets` folder next to the script.

```bash
python simulate_agents_phase1.py   # Phase 1
python simulate_agents_phase2.py   # Phase 2 (uses dataset12)
python simulate_agents_phase3.py   # Phase 3 (uses dataset1)
```

Phase 2 also writes the chosen classifier's validation predictions to `test_pred.txt`.

## Results

### Phase 1

| Market | Wealth |
|--------|--------|
| Fair | $171,979.38 |
| Junk yard | $85,311.13 |
| Fancy | $254,593.96 |

### Phase 2

Beat the baseline agents in 9 of 12 datasets.

### Phase 3

Final wealth with different classifiers swapped into the agent. The code currently uses `BernoulliNB`; the others were tried by changing `self.clf`.

| Classifier | Final Wealth | Good Products |
|------------|--------------|---------------|
| Logistic Regression | $320,175.71 | 645 |
| GaussianNB | $313,625.52 | 669 |
| AdaBoostClassifier | $313,623.56 | 670 |
| KNeighborsClassifier | $312,379.11 | 620 |
| SVC | $311,824.03 | 631 |
| BernoulliNB | $310,468.12 | 707 |
| RandomForestClassifier | $309,632.43 | 643 |
| DecisionTreeClassifier | $270,815.76 | 536 |
| KMeans | $95,660.20 | 497 |
| Baseline: CheapAgent | $243,838.41 | 498 |
| Baseline: RandomAgent | $15,160.33 | 513 |

These numbers come from the original Python 2 runs and may differ with current scikit-learn.

## Author

Saurabh Katkar
