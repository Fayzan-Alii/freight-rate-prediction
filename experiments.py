# Experiments: the evidence behind the choices in pipeline.py
# Run: python experiments.py
#
# Test 1: which columns help?
#   For each month from June to October, the model learns from all earlier months
#   and is tested on that month. Testing 5 times (not once) shows if a result is consistent.
#
# Test 2: can the model handle cities it has never seen?
#   12% of validation loads use a city that is not in the training data.

import pandas as pd
import lightgbm as lgb


# ------------------------------------------------------------
# Load and clean (same fixes as pipeline.py)
# ------------------------------------------------------------
train = pd.read_csv("data/train_test.csv")
train["date"] = pd.to_datetime(train["date"])
train["weight"] = train["weight"].abs()
train["weight"] = train["weight"].fillna(train["weight"].median())
train["market_index"] = train["market_index"].fillna(train["market_index"].median())
train["equipment_code"] = train["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})
train["day_of_week"] = train["date"].dt.dayofweek
train["day_of_month"] = train["date"].dt.day
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
train["is_normal"] = (train["rate_per_mile"] >= 1.5) & (train["rate_per_mile"] <= 4)


def new_model():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, objective="l1", random_state=0, verbose=-1)


# ------------------------------------------------------------
# Test 1: which columns help?
# ------------------------------------------------------------
chosen = ["distance", "weight", "equipment_code", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "day_of_week"]
coordinates = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

experiments = {
    "chosen features":    chosen,
    "add quote_signal":   chosen + ["quote_signal"],
    "add market_index":   chosen + ["market_index"],
    "add day_of_month":   chosen + ["day_of_month"],
    "remove day_of_week": ["distance", "weight", "equipment_code"] + coordinates,
    "remove coordinates": ["distance", "weight", "equipment_code", "day_of_week"],
    "remove weight":      ["distance", "equipment_code", "day_of_week"] + coordinates,
}

print("TEST 1: average dollar error per test month (lower is better)")

for name, features in experiments.items():
    errors = []

    for month in [6, 7, 8, 9, 10]:
        learn = train[(train["date"].dt.month < month) & train["is_normal"]]
        test = train[train["date"].dt.month == month].copy()

        # In validation, quote_signal has no link to price (about 2.05 for every load).
        # Mixing up its order in the test month makes the test look like validation.
        test["quote_signal"] = test["quote_signal"].sample(frac=1, random_state=0).values

        model = new_model()
        model.fit(learn[features], learn["rate_per_mile"])
        predicted = model.predict(test[features]) * test["distance"]

        error = (test["posted_rate"] - predicted).abs().mean()
        errors.append(round(float(error), 1))

    average = round(sum(errors) / len(errors), 1)
    print(name, "| Jun-Oct:", errors, "| average:", average)


# ------------------------------------------------------------
# Test 2: cities the model has never seen
# Hide 8 cities (picked at random) while the model learns from Jan-Aug.
# Then compare its error on Sep-Oct loads with and without those cities.
# ------------------------------------------------------------
hidden_cities = ["St. Louis", "Hartford", "Amarillo", "Toledo", "Richmond", "Birmingham", "New York", "Jacksonville"]

train["uses_hidden_city"] = train["pickup"].isin(hidden_cities) | train["delivery"].isin(hidden_cities)

learn = train[(train["date"] < "2025-09-01") & train["is_normal"] & ~train["uses_hidden_city"]]
test = train[train["date"] >= "2025-09-01"].copy()

model = new_model()
model.fit(learn[chosen], learn["rate_per_mile"])
test["error"] = (test["posted_rate"] - model.predict(test[chosen]) * test["distance"]).abs()

known_error = round(test[~test["uses_hidden_city"]]["error"].mean(), 1)
hidden_error = round(test[test["uses_hidden_city"]]["error"].mean(), 1)

print()
print("TEST 2: hidden cities:", hidden_cities)
print("  error on loads with only known cities: $", known_error)
print("  error on loads with a hidden city:     $", hidden_error)
