@echo off
REM run_training.bat — Train the acne severity classifier
echo Training EfficientNet-B0 Acne Classifier...
cd /d "d:\Anti Gravity\ML Project"
.venv\Scripts\python.exe ML/training/train_acne_classifier.py --epochs 20 --batch_size 32 --lr 1e-4 --patience 7
pause
