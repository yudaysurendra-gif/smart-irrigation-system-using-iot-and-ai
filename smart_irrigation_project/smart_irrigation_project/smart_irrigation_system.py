"""
==========================================================================
 SMART IRRIGATION SYSTEM USING IoT AND AI
==========================================================================
A simulation project that demonstrates how IoT sensors (soil moisture,
temperature, humidity, rainfall) can feed data into an AI model
(Random Forest Classifier) that predicts whether irrigation (the pump)
should be turned ON or OFF.

Since no physical hardware is connected, this script SIMULATES:
    1. IoT Sensor Nodes  -> generate realistic sensor readings
    2. Cloud/Edge AI      -> a trained ML model decides irrigation need
    3. Actuator (Pump)    -> ON/OFF decision is logged and "executed"
    4. Data Logger        -> stores every reading + decision to a CSV file

To connect this to REAL hardware later, simply replace the
`IoTSensorSimulator.read_sensors()` method with actual sensor
reads (e.g., using RPi.GPIO / Adafruit DHT22 / capacitive soil sensor
libraries), and replace `Actuator.set_pump()` with a relay/GPIO call.

Author: (Your Name)
==========================================================================
"""

import random
import time
import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================================================
# 1. IoT SENSOR SIMULATION
# ==========================================================================
class IoTSensorSimulator:
    """
    Simulates a set of IoT sensors deployed in the field:
        - Soil Moisture (%)      : lower value = drier soil
        - Temperature (°C)
        - Humidity (%)
        - Rainfall probability (mm, simulated forecast/rain sensor)
    In a real deployment these would come from sensors such as
    capacitive soil moisture sensors, DHT22, and a rain gauge,
    transmitted over WiFi/LoRa/MQTT to a central controller.
    """

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Start with a "typical" field state
        self.soil_moisture = random.uniform(30, 60)
        self.temperature = random.uniform(20, 35)
        self.humidity = random.uniform(30, 70)
        self.rainfall = 0.0

    def read_sensors(self):
        """Simulate the next sensor reading with small random fluctuations."""
        # Soil naturally dries out a bit each cycle unless it rains
        self.soil_moisture += random.uniform(-4, 1.5)
        self.temperature += random.uniform(-1.5, 1.5)
        self.humidity += random.uniform(-3, 3)

        # Occasionally simulate rainfall
        self.rainfall = random.choice([0, 0, 0, 0, random.uniform(1, 15)])
        if self.rainfall > 0:
            self.soil_moisture += self.rainfall * 0.8
            self.humidity += 10

        # Clamp values to realistic physical ranges
        self.soil_moisture = float(np.clip(self.soil_moisture, 0, 100))
        self.temperature = float(np.clip(self.temperature, 5, 45))
        self.humidity = float(np.clip(self.humidity, 10, 100))

        return {
            "soil_moisture": round(self.soil_moisture, 2),
            "temperature": round(self.temperature, 2),
            "humidity": round(self.humidity, 2),
            "rainfall": round(self.rainfall, 2),
        }


# ==========================================================================
# 2. AI MODEL — DECIDES WHETHER TO IRRIGATE
# ==========================================================================
class IrrigationAIModel:
    """
    A Random Forest Classifier trained on synthetic (rule-based) historical
    data to learn the relationship between sensor readings and the
    correct irrigation decision. In a real system, this training data
    would come from historical logs / agronomist-labeled data.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        )
        self._train()

    @staticmethod
    def _label_rule(soil_moisture, temperature, humidity, rainfall):
        """
        Ground-truth rule used ONLY to generate synthetic training labels.
        Irrigate (1) when soil is dry, it's hot, humidity is low,
        and it isn't currently raining.
        """
        if rainfall > 2:
            return 0
        if soil_moisture < 35 and humidity < 60:
            return 1
        if soil_moisture < 25:
            return 1
        if soil_moisture > 55:
            return 0
        # Borderline cases: hot & dry-ish tips it towards irrigation
        if temperature > 32 and soil_moisture < 45:
            return 1
        return 0

    def _generate_synthetic_dataset(self, n=4000):
        data = []
        for _ in range(n):
            soil = random.uniform(0, 100)
            temp = random.uniform(5, 45)
            hum = random.uniform(10, 100)
            rain = random.choice([0, 0, 0, random.uniform(0, 20)])
            label = self._label_rule(soil, temp, hum, rain)
            data.append([soil, temp, hum, rain, label])
        df = pd.DataFrame(
            data,
            columns=["soil_moisture", "temperature", "humidity", "rainfall", "irrigate"],
        )
        return df

    def _train(self):
        df = self._generate_synthetic_dataset()
        X = df[["soil_moisture", "temperature", "humidity", "rainfall"]]
        y = df["irrigate"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[AI MODEL] Training complete. Validation accuracy: {acc * 100:.2f}%\n")

    def predict(self, reading: dict):
        """Return (decision, confidence) where decision is True/False."""
        features = pd.DataFrame([{
            "soil_moisture": reading["soil_moisture"],
            "temperature": reading["temperature"],
            "humidity": reading["humidity"],
            "rainfall": reading["rainfall"],
        }])
        proba = self.model.predict_proba(features)[0]
        decision = bool(self.model.predict(features)[0])
        confidence = proba[1] if decision else proba[0]
        return decision, round(confidence * 100, 1)


# ==========================================================================
# 3. ACTUATOR — CONTROLS THE WATER PUMP
# ==========================================================================
class Actuator:
    """Simulates a relay-controlled water pump (or solenoid valve)."""

    def __init__(self):
        self.pump_on = False

    def set_pump(self, turn_on: bool):
        self.pump_on = turn_on
        state = "ON 💧" if turn_on else "OFF"
        print(f"[ACTUATOR] Water Pump -> {state}")


# ==========================================================================
# 4. DATA LOGGER
# ==========================================================================
class DataLogger:
    def __init__(self, filepath="irrigation_log.csv"):
        self.filepath = filepath
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "soil_moisture", "temperature",
                    "humidity", "rainfall", "decision", "confidence(%)"
                ])

    def log(self, reading, decision, confidence):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                reading["soil_moisture"],
                reading["temperature"],
                reading["humidity"],
                reading["rainfall"],
                "IRRIGATE" if decision else "SKIP",
                confidence,
            ])


# ==========================================================================
# 5. MAIN CONTROL LOOP — TIES EVERYTHING TOGETHER
# ==========================================================================
def run_smart_irrigation_system(cycles=15, delay_seconds=0.4):
    print("=" * 70)
    print(" SMART IRRIGATION SYSTEM — IoT + AI SIMULATION STARTING")
    print("=" * 70 + "\n")

    sensors = IoTSensorSimulator(seed=7)
    ai_model = IrrigationAIModel()
    pump = Actuator()
    logger = DataLogger()

    for cycle in range(1, cycles + 1):
        print(f"--- Cycle {cycle}/{cycles} ---")
        reading = sensors.read_sensors()
        print(f"[SENSORS] Soil Moisture: {reading['soil_moisture']}% | "
              f"Temp: {reading['temperature']}°C | "
              f"Humidity: {reading['humidity']}% | "
              f"Rainfall: {reading['rainfall']}mm")

        decision, confidence = ai_model.predict(reading)
        print(f"[AI MODEL] Decision: {'IRRIGATE' if decision else 'SKIP'} "
              f"(confidence: {confidence}%)")

        pump.set_pump(decision)
        logger.log(reading, decision, confidence)
        print()

        time.sleep(delay_seconds)

    print("=" * 70)
    print(f" Simulation finished. Full log saved to: {logger.filepath}")
    print("=" * 70)


if __name__ == "__main__":
    run_smart_irrigation_system(cycles=15, delay_seconds=0.3)
