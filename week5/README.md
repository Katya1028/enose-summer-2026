## QCM Data Analysis Pipeline
This project provides an automated pipeline for QCM nose data analysis.

The pipeline can:
- Load raw QCM csv files
- Extract features from 16 QCM channels
- Perform PCA visualization
- Train and evaluate LDA, SVM, and RandomForest models
- Use GroupKFold tp prevent measurements from the same biological sample from appearing in both training and testing sets
- Save analysis results and plots automatically

## Project Structure
project/
|--All/: stores raw QCM csv files grouped by class
|  |--HC/
|  |--adenoma/
|  |--CRC/
|--outputs/: stores generated results
|--data_io.py: handles data loading and file searching
|--features.py: extracts QCM features
|--models.py: performs PCA, model training and evaluation
|--run_pipeline.py: runs the complete analysis pipeline
|--README.md

## Requirements
pandas/ numpy/ matplotlib/ scikit-learn

## Input Data
- Each raw csv files should contain: datetime/ step/ ch1~ch16
- The "step" column identifies different measurement stages such as: Measurement/ Clean
- File naming format: sampleID_bottle_measurement.csv
    Example: "E350_2_1.csv"
    E340 = biological sample ID
    2 = bottle number
    1 = measurement number
    
## How to Run
1. Place raw csv files under "All"
2. run python run_pipeline.py
3. The results will be generated in the "output" folder

## Outputs
- all-features.csv: extracted feature table
- pca.png: 2D PCA visualization
- model_results.csv: cross-validation results for all models
-confusion_matrix.png: confusion matrix of the best model
-classification_report.csv: precision, recall, and F1-score of the best model

## Notes
- Each QCM measurement contains 16 channels
- 5 features are extracted from each channels:delta_f/ response_slope/ recovery_slope/ auc/ t90
- A total 0f 80 features are used for classification