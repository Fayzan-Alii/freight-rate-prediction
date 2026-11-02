# Freight rate prediction pipeline
# Run: python pipeline.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

# The columns the model learns from. Each choice is tested in experiments.py.
# Not used:
#   load_id         - just a label
#   pickup/delivery - city names; coordinates say the same thing and also work for new cities
#   quote_signal    - in validation it has no link to price (about 2.05 for every load)
#   market_index    - made results worse on most test months
#   month           - the model never saw November or December
#   day_of_month    - made results worse
FEATURES = [
    "distance",
    "weight",
    "equipment_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "day_of_week",
]


def new_model():
    # objective="l1" = the model tries to make the average dollar error as small as possible
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, objective="l1", random_state=0, verbose=-1)


def clean(df, typical_weight):
    # Same fixes for training, validation and December data
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].abs()                       # negative weight = minus-sign typo
    df["weight"] = df["weight"].fillna(typical_weight)      # fill missing weights
    df["equipment_code"] = df["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})
    df["day_of_week"] = df["date"].dt.dayofweek             # 0 = Monday ... 6 = Sunday
    return df


def show_error(name, actual, predicted):
    dollars = (actual - predicted).abs().mean()
    percent = ((actual - predicted).abs() / actual).mean() * 100
    print(name, "-> average error: $", round(dollars, 2), "(", round(percent, 2), "% )")


# ------------------------------------------------------------
# Step 1: Load the data
# ------------------------------------------------------------
train = pd.read_csv("data/train_test.csv")
val = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation_predictions_template.csv")
december = pd.read_csv("data/december_chart_inputs.csv")


# ------------------------------------------------------------
# Step 2: Clean
# ------------------------------------------------------------
typical_weight = train["weight"].abs().median()
train = clean(train, typical_weight)
val = clean(val, typical_weight)

# The model learns price per mile, so short and long loads are on the same scale.
# Later we multiply by distance to get the price back.
train["rate_per_mile"] = train["posted_rate"] / train["distance"]

# Real prices sit between $1.50 and $4 per mile, with empty gaps on both sides
# (see charts/01_price_per_mile.png). Loads outside that range look like data errors.
train["is_normal"] = (train["rate_per_mile"] >= 1.5) & (train["rate_per_mile"] <= 4)
print("Outliers found:", (~train["is_normal"]).sum())


# ------------------------------------------------------------
# Step 3: Test the model on a time split
# Learn from Jan-Aug, test on Sep-Oct (like the real task: predict months it hasn't seen)
# ------------------------------------------------------------
learn = train[(train["date"] < "2025-09-01") & train["is_normal"]]   # outliers removed
test = train[train["date"] >= "2025-09-01"]                          # every row kept

print()
print("TIME SPLIT: learn on", len(learn), "loads, test on", len(test), "loads")

# A simple guess the model has to beat: typical price per mile for each trailer type x distance
median_rate = learn.groupby("equipment")["rate_per_mile"].median()
show_error("Baseline: median $/mile by equipment", test["posted_rate"], test["equipment"].map(median_rate) * test["distance"])

model = new_model()
model.fit(learn[FEATURES], learn["rate_per_mile"])
test = test.copy()
test["predicted"] = model.predict(test[FEATURES]) * test["distance"]
show_error("LightGBM model", test["posted_rate"], test["predicted"])

normal_test = test[test["is_normal"]]
show_error("LightGBM model, normal test loads only", normal_test["posted_rate"], normal_test["predicted"])


# ------------------------------------------------------------
# Step 4: Random split, only to compare with the time split
# ------------------------------------------------------------
random_learn, random_test = train_test_split(train, test_size=0.2, random_state=0)
random_learn = random_learn[random_learn["is_normal"]]

model = new_model()
model.fit(random_learn[FEATURES], random_learn["rate_per_mile"])
print()
show_error("RANDOM SPLIT: LightGBM model", random_test["posted_rate"], model.predict(random_test[FEATURES]) * random_test["distance"])


# ------------------------------------------------------------
# Step 5: Train the final model on all of Jan-Oct
# ------------------------------------------------------------
final_train = train[train["is_normal"]]
final_model = new_model()
final_model.fit(final_train[FEATURES], final_train["rate_per_mile"])
print()
print("Final model trained on", len(final_train), "loads")


# ------------------------------------------------------------
# Step 6: Predict the 12,000 validation loads
# ------------------------------------------------------------
val["predicted_rate"] = (final_model.predict(val[FEATURES]) * val["distance"]).round(2)

# Put the predictions into the template, matching rows by load_id
submission = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id")
submission.to_csv("validation_predictions.csv", index=False)

print("Saved validation_predictions.csv")
print("  rows:", len(submission), "| empty:", submission["predicted_rate"].isna().sum(),
      "| zero or negative:", (submission["predicted_rate"] <= 0).sum())


# ------------------------------------------------------------
# Step 7: Predict the 31 December days
# The December file has no coordinates, so we take them from training.
# ------------------------------------------------------------
december_features = clean(december.copy(), typical_weight)

# Each city always has the same coordinates, so take them from any training row
lexington = train[train["pickup"] == "Lexington"].iloc[0]
fort_wayne = train[train["delivery"] == "Fort Wayne"].iloc[0]
december_features["pickup_lat"] = lexington["pickup_lat"]
december_features["pickup_lon"] = lexington["pickup_lon"]
december_features["delivery_lat"] = fort_wayne["delivery_lat"]
december_features["delivery_lon"] = fort_wayne["delivery_lon"]

# Keep the original 7 columns and only fill predicted_rate (the scorer checks this)
december["predicted_rate"] = (final_model.predict(december_features[FEATURES]) * december_features["distance"]).round(2)
december.to_csv("december_predictions.csv", index=False)
print("Saved december_predictions.csv")
