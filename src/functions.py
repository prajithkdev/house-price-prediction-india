"""
Reconstructed helper module for the House Price India notebook.

The original `functions.py` was maintained on a teammate's personal machine during
a team assignment and was never part of this notebook's folder — it isn't
recoverable. This file is a from-scratch reimplementation, written by inspecting
the notebook's own saved outputs (printed metrics, tuple shapes, dataframe
columns) rather than the original source, so that the notebook can actually run.

Known limitation: GridSearchCV results depend on the exact parameter grid,
scoring metric, and cross-validation folds used originally, none of which are
fully recoverable from printed output alone. Re-running this notebook with this
module will reproduce the same *kind* of result (same models, same general
magnitude) but will not necessarily reproduce the exact numbers already saved
in the notebook and reported in the README.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# Maps the raw "House Price India" CSV columns to the snake_case names used
# throughout the notebook. The one entry that keeps its parenthesis
# ("area_of_the_house(excluding_basement)") is intentional: the notebook
# fixes it up separately in a follow-up `df.rename(...)` call.
rename_columns = {
    "number of bedrooms": "number_of_bedrooms",
    "number of bathrooms": "number_of_bathrooms",
    "living area": "living_area",
    "lot area": "lot_area",
    "number of floors": "number_of_floors",
    "waterfront present": "waterfront_present",
    "number of views": "number_of_views",
    "condition of the house": "condition_of_the_house",
    "grade of the house": "grade_of_the_house",
    "Area of the house(excluding basement)": "area_of_the_house(excluding_basement)",
    "Area of the basement": "area_of_the_basement",
    "Built Year": "built_year",
    "Renovation Year": "renovation_year",
    "Postal Code": "postal_code",
    "Lattitude": "lattitude",
    "Longitude": "longitude",
    "Number of schools nearby": "number_of_schools_nearby",
    "Distance from the airport": "distance_from_the_airport",
    "Price": "price",
}


def read_csv(path):
    """Thin pass-through so the notebook's `read_csv(...)` calls resolve."""
    return pd.read_csv(path)


def EDA(df):
    """Rename to snake_case, print a quick summary, and return (df, dtypes, info)."""
    renamed = df.rename(columns=rename_columns)
    print("You can view the exploratory Data analysis here:")
    print(renamed.describe())
    print("Information of the Dataframe is found here:")
    print(renamed.info())
    return renamed, renamed.dtypes, renamed.info


def histogram(series):
    sns.histplot(series, kde=True)
    plt.title(f"Distribution of {series.name}")
    plt.show()


def boxplot(x, y):
    sns.boxplot(x=x, y=y)
    plt.title(f"{y.name} by {x.name}")
    plt.show()


def scatter(x, y):
    sns.scatterplot(x=x, y=y)
    plt.title(f"{y.name} vs {x.name}")
    plt.show()


def line(x, y):
    sns.lineplot(x=x, y=y)
    plt.title(f"{y.name} over {x.name}")
    plt.show()


def _split(df, features, target, test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def decisiontreemodel(df, features, target):
    """Grid-searched decision tree, matching the notebook's winning params
    (max_depth=6, min_samples_split=2) as one point in the search grid."""
    X_train, X_test, y_train, y_test = _split(df, features, target)

    param_grid = {"max_depth": [4, 6, 8, 10, None], "min_samples_split": [2, 5, 10]}
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42), param_grid, scoring="r2", cv=5
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("Best hyperparameters: ", grid.best_params_)
    print("Best RMSE score: ", grid.best_score_)  # actually the R² from `scoring="r2"`
    print("Root Mean Squared Error:", rmse)
    return grid.best_estimator_


def decisiontree(df, features, target, max_depth=6):
    """Fixed-depth decision tree — renders a tree diagram, no metric printed
    (matches the original notebook's saved output for this cell)."""
    X_train, X_test, y_train, y_test = _split(df, features, target)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=features, filled=True, fontsize=8)
    plt.show()
    return model


def xgboost(df, features, target):
    X_train, X_test, y_train, y_test = _split(df, features, target)

    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 300, 500],
    }
    grid = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("Best Parameters:", grid.best_params_)
    print("Best Score:", -grid.best_score_)
    print("Root Mean Squared Error:", rmse)

    importances = pd.Series(grid.best_estimator_.feature_importances_, index=features)
    importances.sort_values().plot(kind="barh", figsize=(10, 8))
    plt.title("XGBoost Feature Importance")
    plt.show()
    return grid.best_estimator_


def random_forest(df, features, target):
    X_train, X_test, y_train, y_test = _split(df, features, target)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error:", rmse)

    importances = pd.Series(model.feature_importances_, index=features)
    importances.sort_values().plot(kind="barh", figsize=(10, 5))
    plt.title("Random Forest Feature Importance")
    plt.show()
    return model
