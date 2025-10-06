#!/bin/bash
echo "Cleaning previous data, models, and reports..."
rm -rf data/features/* data/predictions/* data/models/* reports/*
echo "Building NFL Game Outcome Models..."
python -m src.cli build-all
echo "Done!"