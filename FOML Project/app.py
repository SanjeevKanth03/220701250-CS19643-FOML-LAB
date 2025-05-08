import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
df = pd.read_csv('improved_supply_chain_data.csv')

le_event = LabelEncoder()
le_weather = LabelEncoder()
le_location = LabelEncoder()

df['event_type'] = le_event.fit_transform(df['event_type'])
df['weather_severity'] = le_weather.fit_transform(df['weather_severity'])
df['location'] = le_location.fit_transform(df['location'])

X = df[['event_type', 'weather_severity', 'location']]
y = df['disruption']

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Tkinter UI
root = tk.Tk()
root.title("Supply Chain Disruption Predictor")
root.geometry("400x300")

tk.Label(root, text="Event Type").pack()
event_cb = ttk.Combobox(root, values=le_event.classes_.tolist())
event_cb.pack()

tk.Label(root, text="Weather Severity").pack()
weather_cb = ttk.Combobox(root, values=le_weather.classes_.tolist())
weather_cb.pack()

tk.Label(root, text="Location").pack()
location_cb = ttk.Combobox(root, values=le_location.classes_.tolist())
location_cb.pack()

def predict_disruption():
    try:
        event = le_event.transform([event_cb.get()])[0]
        weather = le_weather.transform([weather_cb.get()])[0]
        location = le_location.transform([location_cb.get()])[0]

        input_data = np.array([[event, weather, location]])
        prediction = rf.predict(input_data)[0]
        proba = rf.predict_proba(input_data)[0]

        if prediction == 1:
            message = f"⚠️ Disruption likely!\n\nProbability: {proba[1]:.2f}"
            messagebox.showerror("Prediction Result", message)
        else:
            message = f"✅ No disruption expected.\n\nProbability: {proba[0]:.2f}"
            messagebox.showinfo("Prediction Result", message)
    except Exception as e:
        messagebox.showwarning("Input Error", "Please select all fields correctly.")

tk.Button(root, text="Predict", command=predict_disruption).pack(pady=20)

root.mainloop()
