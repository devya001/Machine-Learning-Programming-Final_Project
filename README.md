# Breast Cancer Classification using Random Forest and Logistic Regression



## Team Members

- *Khushal Bhalala – Student ID: 200619365*  
- *Devyakumar Patel – Student ID: 200635167*  

---

## Submission Details

- *Course:* AIDI 1002 – Machine Learning Programming  
- *Instructor:* Garima Malik  
- *Institution:* Georgian College


## Project Overview :

*Dataset Used:*  
  Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)  
  [Dataset Link – https://github.com/toandaominh1997/dataset-for-beginners/blob/master/breast_cancer_wisconsin/data.csv] 

---

Summary :
This project builds a binary classification system to detect breast cancer as Malignant (cancerous) or Benign (non-cancerous). We use two machine learning models – a Logistic Regression classifier and a Random Forest classifier – to predict tumor diagnosis based on measurements of cell nuclei from biopsy data. We evaluate both models with multiple metrics to compare their performance. The goal is to achieve high accuracy in distinguishing malignant tumors from benign ones, which can aid in early diagnosis and treatment. This project was developed as the final project for the AIDI 1002 Machine Learning Programming course (Summer 2025).

Features :
Binary Classification Focus: The project classifies tumors into two classes (malignant vs. benign) using supervised learning on labeled data.
Multiple Algorithms: It implements and compares two approaches (logistic regression and random forest) to observe differences in performance.
Imbalanced Data Handling: Techniques such as class weighting (and optionally oversampling via imbalanced-learn) are used to address the imbalance in class distribution (more benign cases than malignant).
Comprehensive Evaluation: Model performance is measured with various metrics – Accuracy, Precision, Recall, F1 Score, and ROC-AUC – for a thorough assessment.
Result Visualization: The project generates visualizations including a confusion matrix for predictions, an ROC curve for classifier performance, and a feature importance plot to interpret which features influence the Random Forest model.


Algorithms Used :
Logistic Regression: A simple linear model used as a baseline classifier. It finds a linear decision boundary to separate the two classes. We applied feature scaling (standardization) before training to ensure the logistic regression converges and performs optimally. Logistic Regression is intuitive and provides interpretable coefficients (weights for each feature).

Random Forest Classifier: An ensemble of many decision trees, providing a more powerful nonlinear classifier. Each decision tree votes on the class, and the forest’s majority vote is the final prediction. Random Forests can capture complex relationships and tend to improve accuracy and robustness. They also offer feature importance estimates, helping us identify which attributes of the data are most influential.
# Breast Cancer Classification – Wisconsin Diagnostic Dataset (WDBC)

---

## Files in This Repository
  [Git Repo Link – https://github.com/devya001/Machine-Learning-Programming-Final_Project] 

| File | Description |
|------|-------------|
| AIDI_1002_Final_Project_WDBC.ipynb | Main notebook: preprocessing, training, evaluation |
| aidi_1002_final_project_wdbc.py | Python script for VS Code execution |
| data.csv | Breast Cancer Wisconsin dataset (diagnostic) |
| Final_project_Report.ipynb | Final report with analysis and results |
| README.md | Project summary and instructions |

---

#How to Run
Get the files: Download or clone the project repository, which contains the Jupyter Notebook (e.g. AIDI_1002_Final_Project_WDBC.ipynb) and the dataset file (wdbc.csv or data.csv). Make sure the CSV dataset is in the same folder as the notebook, or update the notebook path to where the data is located.

Install dependencies: Install the required Python libraries if not already installed (see Software Requirements above). For example, run pip install -U pandas scikit-learn imbalanced-learn matplotlib in your environment.

Open the Notebook: Launch Jupyter Notebook or use an environment like Google Colab. Open the AIDI_1002_Final_Project_WDBC.ipynb file.

Run the notebook: Execute the cells in order (for Jupyter, select Cell -> Run All). The notebook will load the dataset, preprocess the data (e.g., drop unnecessary columns like ID, handle diagnosis labels as 0/1, and apply scaling for logistic regression), then train the Logistic Regression and Random Forest models on the training set.
View outputs: As the cells run, the notebook will output the performance metrics for each model (accuracy, precision, recall, F1, ROC-AUC). It will also display plots for the Confusion Matrix, ROC Curve, and Feature Importance. These help in interpreting the results.
Analyze results: Check which model performed better and how the metrics compare. For example, observe the confusion matrix to see misclassifications, and use the feature importance chart to understand which features were most significant in predicting malignancy. The ROC curve will show how well the model separates the classes across different thresholds (a curve closer to the top-left indicates better performance).

Output : (WHAT YOU GET)
After running the project, you will see various outputs that summarize the model results:

Figure: Example Confusion Matrix for the classification results (Logistic Regression model on the test set). This 2x2 matrix compares the model’s predictions with the actual labels. In this example, out of 143 test samples, the model correctly predicted 89 benign cases and 50 malignant cases. There was 1 false alarm (one benign misclassified as malignant, shown as a false positive) and 3 missed detections (three malignant cases misclassified as benign, shown as false negatives). A perfect classifier would have only diagonal entries (all true positives and true negatives), and as we can see, this model performs very well with only a few errors.

Figure: ROC Curve (Receiver Operating Characteristic curve) for the trained models. The ROC curve illustrates the trade-off between the true positive rate (sensitivity) and false positive rate for different classification thresholds. Here we show the ROC curves for both Logistic Regression and Random Forest classifiers on the test data. Both curves hug the top-left corner, indicating high performance (the area under the curve, ROC-AUC, is about 0.99 for both models). A ROC-AUC of 1.0 represents a perfect classifier
mdpi.com
, so our models are nearly perfect at distinguishing malignant vs. benign cases on this dataset.

Figure: Feature Importance plot from the Random Forest model. This bar chart ranks the top 10 most important features in making the breast cancer predictions. Features related to the size and shape of cell nuclei were most influential. For instance, the “concave points (worst)”, “area (worst)”, “radius (worst)”, and “perimeter (worst)” are among the highest contributors (these correspond to characteristics of the cell nucleus for the worst or largest instance in each tumor). High importance of these features suggests that larger and more irregular cell nucleus shapes are strong indicators of a malignant tumor. By contrast, features with lower importance (not shown in the top 10) had less influence on the model’s decisions. This insight can help medical researchers focus on the most diagnostic measurements for breast cancer.

Along with these plots, the notebook prints out the evaluation metrics for each model. For example, both the Logistic Regression and Random Forest achieved an accuracy around 97% on the test set, with very high precision and recall scores (indicating few false positives and few false negatives, respectively). Such performance demonstrates that the models can effectively distinguish between benign and malignant cases on the WDBC dataset.

---
References :
UCI Machine Learning Repository – Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
archive.ics.uci.edu
---
mdpi.com
 (original dataset source and description).
 ---
Kaggle – Breast Cancer Wisconsin (Diagnostic) Data Set (WDBC), Online Dataset.  / 
 https://github.com/devya001/Machine-Learning-Programming-Final_Project
---
