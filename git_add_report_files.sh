#!/bin/bash

echo "Adding essential files for report..."

# Reset everything first
git reset HEAD .

# Add all Python source code (excluding tests and cache)
echo "Adding Python source code..."
git add models/*.py
git add models/**/*.py
git add training/*.py
git add training/**/*.py
git add evaluation/*.py
git add utils/*.py
git add hyperparameter_tuning.py

# Add notebooks
echo "Adding notebooks..."
git add *.ipynb
git add notebooks/*.ipynb

# Add configuration files
echo "Adding configuration files..."
git add configs/*.json
git add requirements.txt

# Add documentation
echo "Adding documentation..."
git add *.md
git add LICENSE 2>/dev/null

# Add important visualization results
echo "Adding visualization results..."
git add tuning_results/*.png
git add tuning_results/*.csv
git add logs/**/*.png 2>/dev/null

# Add .gitignore itself
git add .gitignore

# Add small processed data if exists (for reproducibility)
if [ -f "data/processed/sst2_train.pkl" ]; then
    FILE_SIZE=$(stat -f%z "data/processed/sst2_train.pkl" 2>/dev/null || stat -c%s "data/processed/sst2_train.pkl" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 10485760 ]; then  # Less than 10MB
        echo "Adding small processed data files..."
        git add data/processed/*.pkl
        git add data/processed/*.json
    fi
fi

echo ""
echo "Files staged for commit:"
echo "========================"
git status --short | grep "^A" | head -20

echo ""
echo "Summary:"
git status --short | grep "^A" | wc -l | xargs echo "Total files staged:"
git status --short | grep "^A.*\.py$" | wc -l | xargs echo "Python files:"
git status --short | grep "^A.*\.ipynb$" | wc -l | xargs echo "Notebooks:"
git status --short | grep "^A.*\.png$" | wc -l | xargs echo "Images/plots:"

echo ""
echo "Ready to commit! Use:"
echo "  git commit -m 'Add transformer comparison implementation with visualizations'"
