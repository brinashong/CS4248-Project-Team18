# Team 18: Intent-Aware Citation Classification and Paper Retrieval System

## Dependency Installation
```
pip install -r requirements.txt
```

## Directory Structure
1. [SciCite](./scicite): contains the original dataset downloaded from https://github.com/allenai/scicite.

2. [SciCite Preprocessed](./scicite_preprocessed): where the processed dataset, retrieval databse and selected classifier will be saved.

3. [Preprocessing](./preprocessing): contains code implementation different feature extaction and selection combinations, as well as to build retrieval database.

4. [Experiments](./experiments): contains code implementation model comparisons, model refinement, selected classifier and document retrieval.

5. [Results](./results): contains the retrieved document results and evaluations.

6. [Appendix](./appendix): contains code implementation of EDA, feature selection, extra preprocessing and extra experiments.

## Instructions to reproduce results presented in our report:
1. Perform data preprocessing and feature extraction on the SciCite dataset by running all the cells in [preprocessing/selected-features.ipynb](preprocessing/selected-features.ipynb). Two csv files should be automatically generated and saved as scicite_preprocessed/train-selected-features.csv and scicite_preprocessed/test-selected-features.csv. 

2. Train the fine-tuned Random Forest Classifier on the processed training dataset by running all the cells in [experiments/selected-classifier.ipynb](experiments/selected-classifier.ipynb). The model's Macro-F1 score on the test dataset is 82.17% and will be saved as scicite_preprocessed/selected-classifer.pkl. 

3. Build the retrieval database by running all the cells in [preprocessing/build-retrieval-database.ipynb](preprocessing/build-retrieval-database.ipynb). Three files should be automatically generated and saved as scicite_preprocessed/train_background.jsonl, scicite_preprocessed/train_method.jsonl and scicite_preprocessed/train_result.jsonl.

3. Retrieve similar citations and their corresponding paper IDs by running all the cells in [experiments/document-retrieval.ipynb](experiments/document-retrieval.ipynb).