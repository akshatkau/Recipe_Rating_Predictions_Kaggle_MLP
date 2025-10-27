# Recipe Rating Predictions – Kaggle MLP Project

### Overview
This project aims to predict **user ratings for food recipes** (on a scale of 0–5) using **machine learning techniques**.  
The data is sourced from the Kaggle competition *“Recipe for Rating: Predict Food Ratings using ML”*.  
The goal is to develop an end-to-end ML pipeline — from **data preprocessing** and **feature engineering** to **model training and evaluation**.

---

### Objective
To build a machine learning system that can:
- Predict how users rate a recipe based on their textual reviews and engagement data  
- Explore relationships between textual sentiment and numerical feedback  
- Identify the most effective model architecture for rating prediction  

---

### Dataset Description
**Files Used:**
- `train.csv` – Training dataset (13,636 entries, 15 columns)  
- `test.csv` – Test dataset (4,546 entries, 14 columns)  
- `sample.csv` – Sample submission format  

**Key Features:**
| Feature Type | Columns |
|---------------|----------|
| Numerical | `UserReputation`, `ReplyCount`, `ThumbsUpCount`, `ThumbsDownCount`, `BestScore` |
| Textual | `Recipe_Review` |
| Temporal | `CreationTimestamp` (used to extract Year, Month, Day, Hour, PartOfDay) |
| Target | `Rating` (integer 0–5) |

---

### Approach
1. **Data Cleaning**
   - Removed/filled missing values in `Recipe_Review`
   - Dropped irrelevant identifiers (`UserID`, `UserName`, `CommentID`)
2. **Feature Engineering**
   - Extracted date-time components (Year, Month, Hour, etc.)
   - Created new categorical feature: `PartOfDay`
3. **Preprocessing**
   - Scaled numerical features using `MinMaxScaler`
   - Encoded categorical features with `OneHotEncoder`
   - Transformed text data using `TF-IDF Vectorizer`
4. **Feature Selection**
   - Applied `SelectKBest` with chi-squared test to retain top 2000 features
5. **Model Training**
   - Implemented and tuned multiple models: Logistic Regression, Random Forest, XGBoost, and LightGBM  
   - Used `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning

---

### Technologies Used
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook, Kaggle, GitHub

---

### How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/akshatkau/Recipe_Rating_Predictions_Kaggle_MLP.git
   cd Recipe_Rating_Predictions_Kaggle_MLP
   ```
2. Launchj the notebook:
   ```bash
   jupyter notebook "21f3000376-notebook-t12024 (5).ipynb"
   ```
   
