import pandas as pd
import numpy as np
import pyttsx3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# -------------------------------------------------------
# 🔹 Load both datasets ONCE (not every time function runs)
# -------------------------------------------------------
main_df = pd.read_excel("Hyderabad_Traffic_Patterns.xlsx")
alerts_df = pd.read_excel("Traffic_Prediction_With_Alerts.xlsx")

main_df = main_df.replace("-", np.nan)
main_df.dropna(inplace=True)

# Datatype fixes
for df in [main_df, alerts_df]:
    df["Source"] = df["Source"].astype(str)
    df["Destination"] = df["Destination"].astype(str)
    if "Event" in df.columns:
        df["Event"] = df["Event"].astype(str)

# Encode categorical columns
encoders = {}
categorical_cols = ["Day", "Time_Slot", "Source", "Destination", "Event"]
for col in categorical_cols:
    if col in main_df.columns:
        enc = LabelEncoder()
        main_df[col] = enc.fit_transform(main_df[col])
        encoders[col] = enc

# Features and target
X = main_df[["Day", "Time_Slot", "Source", "Destination", "Rain(mm)", "Avg_Travel_Time(min)"]]
y = main_df["Traffic_Level"]

label_encoder_y = LabelEncoder()
y_enc = label_encoder_y.fit_transform(y)

# Train model once
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------------
# 🔹 Define reusable function for Flask
# -------------------------------------------------------
def predict_traffic(source_input, dest_input):
    source_input = source_input.strip().capitalize()
    dest_input = dest_input.strip().capitalize()

    alerts_df["Source"] = alerts_df["Source"].astype(str)
    alerts_df["Destination"] = alerts_df["Destination"].astype(str)

    subset = alerts_df[
        (alerts_df["Source"].str.lower() == source_input.lower()) &
        (alerts_df["Destination"].str.lower() == dest_input.lower())
    ]

    if not subset.empty:
        subset = subset.sort_values(by=["Predicted_Traffic", "Avg_Travel_Time(min)"])
        best_row = subset.iloc[0]

        best_slot = best_row["Time_Slot"]
        traffic = best_row["Predicted_Traffic"]
        alert = best_row["Traffic_Alert"]
        condition = best_row["Condition"]
        rain = best_row["Rain(mm)_weather"]
        temperature = best_row["Temperature(°C)"]

    else:
        # Fallback: use model prediction
        def safe_encode(enc, value):
            if value in enc.classes_:
                return enc.transform([value])[0]
            else:
                return np.random.choice(range(len(enc.classes_)))

        time_slots = encoders["Time_Slot"].classes_
        best_option = None
        best_pred = 2  # worst traffic by default

        for slot in time_slots:
            data_point = pd.DataFrame({
                "Day": [encoders["Day"].transform(["Monday"])[0]],
                "Time_Slot": [safe_encode(encoders["Time_Slot"], slot)],
                "Source": [safe_encode(encoders["Source"], source_input)],
                "Destination": [safe_encode(encoders["Destination"], dest_input)],
                "Rain(mm)": [0],
                "Avg_Travel_Time(min)": [main_df["Avg_Travel_Time(min)"].mean()]
            })

            pred = model.predict(data_point)[0]
            if pred < best_pred:
                best_pred = pred
                best_option = slot

        best_slot = best_option or "08:00–09:00"
        traffic = label_encoder_y.inverse_transform([best_pred])[0]
        alert = "No alert"
        condition = "Clear"
        rain = 0
        temperature = 28

    # Helper: get time period
    def get_period(slot):
        try:
            hour = int(slot.split(":")[0])
        except:
            return "daytime"
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    period = get_period(str(best_slot))

    # Friendly slot format
    try:
        start_hour = int(str(best_slot).split(":")[0])
        am_pm = "AM" if start_hour < 12 else "PM"
        start_hour_12 = start_hour if start_hour <= 12 else start_hour - 12
        readable_slot = f"{start_hour_12}:00 {am_pm}"
    except:
        readable_slot = str(best_slot)

    # Natural text output
    if str(traffic).lower() == "low":
        best_time_text = f"Roads are clear — ideal time to start your trip is in the {period}."
    elif str(traffic).lower() == "medium":
        best_time_text = f"Moderate traffic expected — travelling in the {period} should be fine."
    else:
        best_time_text = f"Heavy traffic ahead — it’s better to avoid travelling in the {period} if possible."

    # Voice output
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.say(best_time_text)
    engine.runAndWait()

    # Return final summary for Flask
    return (
        f"🚦 Suggested Travel Plan:<br>"
        f"🛣️ Route: {source_input} → {dest_input}<br>"
        f"🕒 Recommended Slot: {readable_slot} ({period.capitalize()})<br>"
        f"🌦️ Weather: {condition}, Rain: {rain} mm, Temp: {temperature}°C<br>"
        f"🚗 Predicted Traffic: {traffic}<br>"
        f"⚠️ Alert: {alert}<br>"
        f"🔊 {best_time_text}"
    )
