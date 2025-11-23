#!/bin/bash

# Script to clean up git repository from data and large files

echo "Removing data and large files from git tracking..."

# Remove from staging area
git reset HEAD data/ checkpoints/ logs/ tuning_results/ __pycache__/ 2>/dev/null
git reset HEAD **/*.pkl **/*.pt **/*.pth **/*.bin **/__pycache__/ 2>/dev/null
git reset HEAD **/*.png **/*.jpg **/*.jpeg 2>/dev/null

# If files were already committed, remove them from repository but keep locally
# Uncomment these lines if needed:
# git rm -r --cached data/ 2>/dev/null
# git rm -r --cached checkpoints/ 2>/dev/null
# git rm -r --cached logs/ 2>/dev/null
# git rm -r --cached tuning_results/ 2>/dev/null
# git rm -r --cached __pycache__/ 2>/dev/null
# git rm --cached **/*.pkl 2>/dev/null
# git rm --cached **/*.pt 2>/dev/null

echo "Done! Current git status:"
git status --short | head -20

echo ""
echo "Files/folders now ignored (from .gitignore):"
cat .gitignore | grep -E "^[^#]" | head -10
