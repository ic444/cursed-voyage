
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

FEATURES = ["distance", "waves", "wind"]

# Model 1: predicts the travel time of a route (linear regression).

def train_model(voyages):
    X, y = voyages[FEATURES], voyages["days"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, test_mae


def predict_days(model, distance, waves, wind):
    row = pd.DataFrame([[distance, waves, wind]], columns=FEATURES)
    return float(model.predict(row)[0])


def contributions(model, distance, waves, wind):
    """Days added by each factor: [base, distance, waves, wind]."""
    c = model.coef_
    return [float(model.intercept_), float(c[0] * distance),
            float(c[1] * waves), float(c[2] * wind)]