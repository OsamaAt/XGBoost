California Housing Price Prediction using XGBoost
This project uses the California Housing dataset from Scikit-learn to predict median house prices using the XGBoost Regressor.
Features
- Dataset: fetch_california_housing
- Model: XGBoost Regressor
- Evaluation Metrics: MAE, MSE, RMSE
- Visualization of Feature Importance

Workflow
- Load Data: Using Scikit-learn’s built-in dataset.
- Data Preprocessing:- Split into training/testing.
- Scale the features using StandardScaler.

- Modeling:- Train using XGBoostRegressor with 200 trees, max_depth=4, and learning_rate=0.1.

- Evaluation:- Print model scores.
- Show actual vs predicted prices.
- Plot feature importances.


Results
- Achieved decent performance on both training and testing datasets.
- Visualized the impact of each feature on the predictions.

Requirements
- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost

Run the code
python your_script.py

Author OsamaAt
--------------------------------------------------------------------------------------------------------------------------------------------------------------
### مشروع توقع أسعار المنازل في كاليفورنيا باستخدام XGBoost

#### **المقدمة**
يهدف هذا المشروع إلى توقع الأسعار المتوسطة للمنازل باستخدام مجموعة بيانات California Housing المتوفرة في مكتبة Scikit-learn، وذلك باستخدام نموذج XGBoost Regressor.

#### **المعالم الرئيسية:**
- **مجموعة البيانات:** `fetch_california_housing`
- **النموذج:** XGBoost Regressor
- **مقاييس التقييم:** MAE، MSE، RMSE
- **تصور أهمية الميزات**

---

### **الخطوات العملية**

#### **تحميل البيانات**
- يتم استخدام مجموعة بيانات مبنية في مكتبة Scikit-learn.

#### **معالجة البيانات**
1. تقسيم البيانات إلى مجموعات تدريب واختبار.
2. قياس الميزات باستخدام `StandardScaler`.

#### **النمذجة**
- تدريب النموذج باستخدام `XGBoostRegressor` بـ:
  - 200 شجرة.
  - أقصى عمق `max_depth=4`.
  - معدل التعلم `learning_rate=0.1`.

#### **التقييم**
- طباعة أداء النموذج.
- عرض مقارنة بين القيم الفعلية والمتوقعة.
- رسم تصور لأهمية الميزات.

---

### **النتائج**
- تحقيق أداء جيد في كل من مجموعتي التدريب والاختبار.
- تصور تأثير كل ميزة على التوقعات.

---

### **المتطلبات البرمجية**
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`

---

### **تنفيذ الكود**
قم بتشغيل الكود باستخدام:
```bash
python your_script.py
```

**المؤلف:** OsamaAt
