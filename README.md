# NBA Salary Predictor

**Machine learning model that predicts NBA player salaries based on performance statistics.**

Built classification models to predict which salary bracket a player falls into using game stats like points, assists, rebounds, and minutes played. Achieved 72% accuracy using a tuned Random Forest classifier.

---

## The Problem

NBA teams and agents need to evaluate fair market value for players during contract negotiations. Traditional methods rely on subjective comparisons and gut feeling.

## The Solution

A data-driven approach that maps on-court performance directly to expected compensation:

```
Player Stats (PTS, AST, REB, MP, etc.)
            │
            ▼
    ┌───────────────┐
    │  Feature      │
    │  Engineering  │
    │  (20 features)│
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  ML Models    │
    │  (KNN, RF)    │
    └───────────────┘
            │
            ▼
    Salary Bracket Prediction (1-10)
```

---

## Technical Approach

### Data Processing
- 500+ NBA players with 30+ statistical features
- Cleaned and encoded categorical variables (team, position)
- Created 10 salary brackets for classification
- Feature selection using correlation threshold (>0.25)

### Final Features Used
```
Age, Games, Games Started, Minutes Played,
Field Goals, 3-Pointers, 2-Pointers, Free Throws,
Rebounds, Assists, Steals, Blocks, Turnovers, Points
```

### Models Evaluated

| Model | Base Accuracy | Tuned Accuracy |
|-------|--------------|----------------|
| K-Nearest Neighbors | 66% | 69% |
| Random Forest | 69% | 72% |

### Hyperparameter Tuning
Used GridSearchCV with 5-fold cross-validation to optimize:
- **KNN**: leaf_size, n_neighbors, distance metric
- **Random Forest**: n_estimators, max_depth, min_samples_split

---

## Tech Stack

- **Python** - Core language
- **pandas** - Data manipulation
- **scikit-learn** - ML models, GridSearchCV, train/test split
- **matplotlib/seaborn** - Visualizations
- **Jupyter Notebook** - Development environment

---

## Key Findings

1. **Points and Minutes** have the highest correlation with salary
2. **Position** has minimal impact on salary prediction
3. **Age** matters less than on-court production
4. **Tuning hyperparameters** improved accuracy by 3-6%

---

## Running the Notebook

```bash
git clone https://github.com/druvsarin1/NBA-Predictive-Model.git
cd NBA-Predictive-Model

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook "DS3000 Final Project.ipynb"
```

---

## Skills Demonstrated

- **Machine Learning** - Classification, model selection, hyperparameter tuning
- **Data Analysis** - Feature engineering, correlation analysis, data cleaning
- **Python** - pandas, scikit-learn, data visualization

---

## Contact

**Druv Sarin** - [LinkedIn](https://linkedin.com/in/druvsarin) | [GitHub](https://github.com/druvsarin1)
