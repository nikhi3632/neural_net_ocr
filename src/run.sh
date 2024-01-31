#!/bin/bash

python3 fully_connected_network.py
python3 train_model.py
python3 optical_character_recognition.py
rm -rf __pycache__