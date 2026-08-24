📦 Prediksi Risiko Keterlambatan Pengiriman Paket

📌 Deskripsi Project

Project ini merupakan implementasi Machine Learning untuk memprediksi risiko keterlambatan pengiriman paket menggunakan dataset DataCo Smart Supply Chain.

Tujuan utama project adalah mengklasifikasikan suatu pesanan menjadi:

- "0" → Tidak Terlambat
- "1" → Terlambat

Prediksi ini diharapkan dapat membantu perusahaan logistik mengidentifikasi pesanan yang berisiko terlambat sehingga tindakan mitigasi dapat dilakukan lebih awal.

---

🎯 Latar Belakang

Ketepatan waktu pengiriman merupakan salah satu faktor penting dalam kepuasan pelanggan. Keterlambatan dapat menyebabkan ketidakpuasan, menurunkan kepercayaan pelanggan, serta berdampak pada operasional dan reputasi perusahaan.

Keterlambatan pengiriman dapat dipengaruhi oleh berbagai faktor seperti mode pengiriman, jadwal pengiriman, wilayah, transaksi, dan karakteristik pesanan.

Oleh karena itu, Machine Learning digunakan untuk mempelajari pola dari data historis pengiriman dan memprediksi apakah suatu pesanan memiliki risiko mengalami keterlambatan.

---

📊 Dataset

Dataset yang digunakan adalah DataCo Smart Supply Chain, yang berisi data transaksi, pelanggan, produk, dan proses pengiriman.

- Jumlah data asli: 180.519 baris
- Jumlah fitur: 53 fitur
- Data yang digunakan dalam penelitian: 40.000 data
- Sampling: Stratified Random Sampling
- Target: "Late_delivery_risk"
- Jenis masalah: Binary Classification
- Jenis Machine Learning: Supervised Learning

Target

- "0" → Tidak Terlambat
- "1" → Terlambat

---

🔧 Preprocessing

Tahapan preprocessing yang dilakukan meliputi:

1. Penanganan missing value
2. Penghapusan fitur yang tidak relevan
3. Penghapusan fitur yang berpotensi menyebabkan data leakage
4. Penghapusan data duplikat
5. Encoding fitur kategorikal
6. Feature engineering
7. Feature selection
8. Standardisasi pada Logistic Regression
9. Train-test split dengan rasio 80:20
10. Penanganan ketidakseimbangan kelas menggunakan SMOTE pada model tertentu.

Fitur yang berpotensi menyebabkan data leakage seperti Days for Shipping (real) dan Delivery Status dihapus karena informasi tersebut baru diketahui setelah proses pengiriman berlangsung.

---

🤖 Algoritma yang Digunakan

1. Logistic Regression

Logistic Regression digunakan sebagai model baseline karena cocok untuk klasifikasi biner dan mampu menghasilkan probabilitas suatu data masuk ke kelas tertentu.

Hyperparameter tuning dilakukan menggunakan GridSearchCV dengan 5-Fold Cross Validation.

Best Parameter:

C = 0.01
class_weight = balanced
penalty = l2
solver = lbfgs

Hasil akurasi: 69%.

---

2. Random Forest

Random Forest digunakan karena mampu menangani hubungan yang kompleks dan non-linear dengan menggabungkan banyak Decision Tree menggunakan pendekatan ensemble learning.

Hyperparameter tuning dilakukan menggunakan RandomizedSearchCV dengan 5-Fold Cross Validation.

Best Parameter:

n_estimators = 200
min_samples_split = 20
min_samples_leaf = 10
max_depth = 12

Hasil akurasi: 71,62%.

---

3. XGBoost

XGBoost digunakan sebagai algoritma ensemble berbasis gradient boosting yang mampu menangani hubungan non-linear dan memperbaiki kesalahan prediksi dari model sebelumnya.

Hyperparameter tuning dilakukan menggunakan RandomizedSearchCV dengan 5-Fold Cross Validation.

Best Parameter:

subsample = 0.9
reg_lambda = 1
reg_alpha = 0.5
n_estimators = 500
min_child_weight = 5
max_depth = 5
learning_rate = 0.05
gamma = 0
colsample_bytree = 0.8

Hasil akurasi: 71,49%.

---

⚙️ Mengapa Menggunakan Hyperparameter Tuning?

Hyperparameter tuning digunakan untuk mencari kombinasi parameter model yang memberikan performa dan generalisasi terbaik.

Hal ini penting karena model baseline Random Forest menunjukkan overfitting yang cukup besar:

- Training Accuracy: 100%
- Testing Accuracy: 71,24%
- Gap: 28,76%

Setelah dilakukan hyperparameter tuning:

- Training Accuracy: 72,90%
- Testing Accuracy: 71,62%
- Gap: hanya 1,27%

Artinya, tuning berhasil mengurangi overfitting secara signifikan dan membuat model lebih mampu melakukan generalisasi terhadap data baru.

Selain itu, seluruh model menggunakan 5-Fold Cross Validation agar pemilihan parameter tidak hanya bergantung pada satu pembagian data dan hasil evaluasi menjadi lebih stabil.

---

📈 Perbandingan Model

Model| Accuracy| ROC-AUC| Keterangan
Logistic Regression| 69%| 73,3%| Paling sederhana, tetapi kurang menangkap hubungan non-linear
Random Forest| 71,63%| 78,7%| Akurasi sedikit lebih tinggi
XGBoost| 71,49%| 78,9%| Model terbaik

XGBoost dipilih sebagai model terbaik bukan karena memiliki akurasi tertinggi, tetapi karena lebih baik dalam mendeteksi kelas terlambat.

XGBoost memiliki:

- Recall tertinggi: 60,3%
- F1-Score tertinggi: 70,9%
- ROC-AUC: 78,9%
- ROC-AUC Cross Validation: 78,2%

Dalam kasus ini, recall menjadi metrik penting karena false negative lebih berisiko: pesanan yang sebenarnya terlambat tetapi diprediksi tidak terlambat dapat membuat perusahaan kehilangan kesempatan untuk melakukan mitigasi lebih awal.

---

🔍 Feature Importance

Berdasarkan hasil analisis, fitur yang paling berpengaruh terhadap risiko keterlambatan adalah:

1. Shipping Mode
2. Days for Shipment (scheduled)

Hasil EDA juga menunjukkan bahwa mode pengiriman First Class dan Second Class memiliki tingkat keterlambatan yang lebih tinggi dibandingkan mode lainnya. Hal ini menunjukkan bahwa faktor operasional pengiriman memiliki pengaruh penting terhadap risiko keterlambatan.

---

🏆 Kesimpulan

Berdasarkan seluruh proses eksperimen, XGBoost menjadi model terbaik untuk memprediksi risiko keterlambatan pengiriman.

Meskipun Random Forest memiliki accuracy sedikit lebih tinggi (71,63%) dibandingkan XGBoost (71,49%), XGBoost memiliki recall, F1-score, ROC-AUC, dan ROC-AUC Cross Validation yang lebih tinggi sehingga lebih sesuai untuk tujuan utama penelitian, yaitu mendeteksi pesanan yang berisiko terlambat.

Hyperparameter tuning juga terbukti penting karena mampu mengurangi overfitting, terutama pada Random Forest, dengan menurunkan gap training-testing accuracy dari 28,76% menjadi 1,27%.

Sementara itu, penggunaan SMOTE dan Voting Classifier tidak memberikan peningkatan performa yang signifikan, sehingga model yang telah dioptimasi melalui hyperparameter tuning sudah cukup baik untuk digunakan sebagai model final.

💡 Rekomendasi

Model XGBoost dapat dikembangkan menjadi sistem early warning untuk membantu perusahaan mengidentifikasi pesanan yang berisiko terlambat sehingga tindakan mitigasi dapat dilakukan sebelum keterlambatan terjadi.

Fokus evaluasi juga dapat diberikan pada Shipping Mode dan jadwal pengiriman, karena kedua faktor tersebut merupakan variabel yang paling berpengaruh terhadap risiko keterlambatan.

---

👩‍💻 Author

Lisa Damayanti
41524010002
S1 Teknik Informatika
Universitas Mercu Buana
2025/2026
