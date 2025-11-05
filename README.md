🚦 Smart Traffic Assistant

A machine learning project that predicts the best travel time based on real-time and historical traffic data. It helps users plan smoother and faster journeys by analyzing traffic patterns, weather alerts, and road conditions.

🧠 What It Does

Predicts traffic levels (Low, Medium, High) for a given route.

Suggests the best time to travel based on past data and alerts.

Gives voice output using text-to-speech for a hands-free experience.

Combines multiple data sources like traffic patterns and road alerts.

⚙️ Tech Stack

Python

Pandas, NumPy – data cleaning and processing

Scikit-learn (RandomForestClassifier) – traffic prediction model

Pyttsx3 – for text-to-speech output

Datetime – for time-based predictions

📂 How It Works

Load and clean traffic datasets (Hyderabad_Traffic_Patterns.xlsx, Traffic_Prediction_With_Alerts.xlsx).

Train a Random Forest Model to predict traffic levels.

Input your current location, destination, and travel time.

The model predicts the traffic level and recommends the best time to travel.

The system speaks out the result for user convenience.

🧾 Example Output
From: Attapur → Yadagirigutta  
Recommended Time: 8:00 AM  
Traffic Level: Medium  
Voice Output: Roads are clear — ideal time to start your trip is in the morning.

🚀 Future Improvements

Integrate Google Maps API for live route data.

Add weather-based traffic prediction.

Build a web interface or mobile app version.
