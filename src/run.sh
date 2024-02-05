#!/bin/bash

echo "Running fully_connected_network.py..."
python3 fully_connected_network.py

echo "Running train_model.py..."
python3 train_model.py

echo "Running optical_character_recognition.py..."
python3 optical_character_recognition.py

echo "Testing run_ocr.py..."
python3 run_ocr.py

echo "Cleaning up __pycache__..."
rm -rf __pycache__

echo "DONE!!!"
