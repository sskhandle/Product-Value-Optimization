# AI Asset Value Optimization

A machine learning system for evaluating product quality and maximizing profit in various market conditions through intelligent buying decisions.

## Project Overview

This project implements an AI agent system that evaluates products based on their features, price, and value to make optimal purchasing decisions. The system was developed through three progressive phases:

1. **Phase I**: A rule-based system using price-to-value ratios to make buying decisions
2. **Phase II**: A machine learning approach with classifier evaluation and selection
3. **Phase III**: An adaptive system that learns from purchased products to improve future decisions

## Features

- **Optimized Decision Algorithm**: Implements a dynamic probability assignment based on price-to-value ratios
- **Classifier Selection System**: Evaluates multiple classifiers using GridSearchCV for parameter tuning
- **Performance Metrics**: Uses accuracy, precision, recall, F1-score, and AUC to determine optimal classifiers
- **Adaptive Learning**: Continuously improves classification by learning from past purchasing decisions
- **Market Adaptation**: Performs well across various market conditions (Fair Market, Junk Yard, Fancy Market)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-product-assessment.git
cd ai-product-assessment

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Phase I: Basic Agent

```python
# Add import statement at the beginning of simulate_agents_phase1.py
from agent_skatkar import Agent_skatkar

# Add the agent to the simulation
agents.append(Agent_skatkar("agent_skatkar"))

# Run the simulation
python simulate_agents_phase1.py
```

### Phase II: Machine Learning Agent

```python
# Add import statement at the beginning of simulate_agents_phase2.py
from agent_skatkar import *

# Add the agent to the simulation
agents.append(Agent_skatkar("My_classifier"))

# Run the simulation
python simulate_agents_phase2.py
```

### Phase III: Adaptive Learning Agent

```python
# Add import statement at the beginning of simulate_agents_phase3.py
from agent_skatkar import *

# Initialize the agent
agent = Agent_skatkar("skatkar")

# Run the simulation
python simulate_agents_phase3.py
```

## Results

### Phase I Performance

The implemented strategy outperformed all baseline agents across different market conditions:

| Market Type | Agent Performance |
|-------------|-------------------|
| Fair Market | $171,979.38 |
| Junk Yard   | $85,311.13 |
| Fancy Market| $254,593.96 |

### Phase II Classifier Performance

Various classification metrics were used to select the best classifier for each dataset, resulting in superior performance compared to baseline agents in 9 out of 12 test cases.

### Phase III Strategy Comparison

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

## Technologies Used

- Python
- Scikit-learn (GridSearchCV, various classification algorithms)
- NumPy/Pandas for data handling
- Anaconda/IPython environment

## Author

Saurabh Katkar
