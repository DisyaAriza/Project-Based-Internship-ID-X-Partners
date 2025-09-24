Credit Risk Prediction

Deskripsi Project:
Proyek ini dibuat sebagai Final Task Project-Based Internship ID/X Partners.
Tujuannya adalah membangun model machine learning untuk memprediksi risiko kredit (credit risk) berdasarkan Loan Dataset, agar perusahaan multifinance dapat meningkatkan akurasi penilaian kelayakan kredit dan mengurangi potensi kredit macet.

Workflow:
Data Understanding – analisis struktur data, distribusi variabel, missing values, dan outlier.
Exploratory Data Analysis (EDA) – analisis univariat & bivariat, korelasi antar fitur, visualisasi pola.
Data Preparation – handling missing value (median/mode), one-hot encoding, scaling (StandardScaler), dan train-test split (80:20) dengan stratifikasi.
Feature Engineering – pemilihan 17 fitur relevan yang paling memengaruhi performa kredit.
Modeling – algoritma: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), dan Random Forest.
Evaluation – metrik: Accuracy, Precision, Recall, F1-score, dan ROC-AUC, dengan visualisasi Confusion Matrix, ROC Curve, dan Classification Report.

Hasil Utama:
Random Forest memberikan performa terbaik dengan ROC-AUC tertinggi dan keseimbangan precision & recall.
Fitur paling berpengaruh: interest rate (int_rate), debt-to-income ratio (dti), dan annual income (annual_inc).
Nasabah dengan bunga rendah, DTI kecil, dan pendapatan tinggi cenderung masuk kategori Good Loan.

Teknologi & Tools:
Python 3 (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Jupyter Notebook / Google Colab
GitHub untuk version control

Cara Menjalankan
Clone repository:
1. git clone https://github.com/DisyaAriza/Project-Based-Internship-ID-X-Partners.git
2. Jalankan Final_Task_Notebook.ipynb untuk analisis lengkap, atau jalankan Final_Task_Code.py untuk pipeline end-to-end.


Project ini dibuat untuk keperluan pembelajaran dan portfolio dalam program Project-Based Internship ID/X Partners.
