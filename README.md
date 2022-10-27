# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
- **Informasi Dataset**
  <br> Dataset yang digunakan pada proyek ini yaitu dataset lengkap dengan prngukuran radiasi matahari selama 4 bulan, informasi lebih lanjut mengenai dataset tersebut dapat lihat pada tabel berikut:

  | Jenis                   | Keterangan                                                                                         |
  | ----------------------- | -------------------------------------------------------------------------------------------------- |
  | Sumber                  | Dataset: [Kaggle](https://www.kaggle.com/datasets/dronio/SolarEnergy?select=SolarPrediction.csv) |
  | Dataset Owner           | ANDREY                                                                                             |
  | Lisensi                 | https://opendatacommons.org/licenses/dbcl/1-0/                                                     |
  | Kategori                | SolarRadiation, Energy                                                                             |
  | Usability               | 8.24                                                                                               |
  | Jenis dan Ukuran Berkas | CSV (2.9 MB)                                                                                       |

  Setelah melakukan observasi pada dataset yang diunduh melalui _link_ Kaggle yaitu `SolarPrediction.csv', didapatkan informasi sebagai berikut :
  
  - Terdapat 32686 baris (_records_ atau jumlah pengamatan) yang berisi informasi mengenai data pengkuran.
  - Terdapat 11 kolom yaitu `UNIXTime, Data, Time, Radiation, Temperature, Pressure, Humidity, WindDirection(Degress), Speed, TimeSunRise, TimeSunSet` yang merupakan variabel - variabel pada data
  - Dari kolom-kolom tersebut terdapat 4 kolom numerik dengan tipe data float64, yaitu `Radiation, Pressure, WindDirection(Degress), Speed` dan terdapat 2 kolom numerik dengan tipe data int64 yaitu `Temperatue, Humidity` yang merupakan fitur numerik. 
  - Terdapat 2 kolom dengan tipe datetime yaitu `UNIXTime, Data, Time, TimeSunRise, TimeSunSet`
  - Tidak terdapat _missing value_ pada dataset. 
  
  Untuk penjelasan mengenai variabel-variabel pada dataset dapat dilihat pada poin-poin berikut ini:

    * UNIXTime = adalah jumlah detik yang telah berlalu sejak 00:00:00 UTC pada 1 Januari 1970 [s]
    * data,Time = Tanggal/waktu saat pengambilan data [%Y-%m-%d %H:%M:%S]
    * Radiation = radiasi yang dipancarkan oleh matahari [W/m^2]
    * Temperatue = Temperature saat pengukuran terjadi [F]
    * Atmospheric pressure = Tekanan atmosfer bumi [Hg]
    * Humidity = Kelembapan saat pengukuran terjadi [%]
    * Wind speed = Kecepatan Angin Pada Saat Pengukuran Terjadi[miles/h]
    * Wind direction = Arah angin yang dilambangkan dengan [degrees]
    * Time SunRise/Sunset = Waktu Matahari terbit dan terbenam [HST(Hawai time)]

- **Sebaran atau Distribusi Data pada Setiap Fitur**
  <br> sebelum masuk ke tahap distribusi data, persiapan yang dilakukan yaitu perlu membuat dua variabel baru yaitu variabel OHLC_Average untuk menampung rata-rata harga dan Price_After_Month untuk harga setelah sebulan.
  <br> Berikut merupakan visualisasi data yang menunjukkan sebaran/distribusi data pada setiap fitur-fitur numerik (`High, Low, Open, Close, OHLC_Average, Price_After_Month`) :
  
  - Mengidentifikasi Missing Value dan Outlier
    <br>
    <image src='https://raw.githubusercontent.com/AzharRizky/Predictive-Anlaytics/main/images/boxplot_outlier.png' width= 500/>
    <br> Terlihat jika di atas banyak terdapat outlier pada setiap variabel, lalu untuk mengatasinya nantinya penulis akan menerapkan batas bawah dan batas atas menggunakan metode IQR
    
  - Univariate Analysis
    <br>
    <image src='https://raw.githubusercontent.com/AzharRizky/Predictive-Anlaytics/main/images/distribusi_data(right-skewed).png' width= 500/>
    <br> Terlihat pada grafik bahwa semua data cenderung distribusi nilainya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model nantinya.
    
  - Multivariate Analysis
    <br>
    <image src='https://raw.githubusercontent.com/AzharRizky/Predictive-Anlaytics/main/images/korelasi_antar_variabel.png' width= 500/>
    <br> Terlihat bahwa pada grafik kebanyakan bernilai positif karena kebanyakan grafik pada sumbu y dan x mengalami peningkatan yang cukup signifikan membentuk sebuah garis lurus.
    
    <br>
    <image src='https://raw.githubusercontent.com/AzharRizky/Predictive-Anlaytics/main/images/corelation_matrix.png' width= 500/>
    <br> Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa semua variabel memiliki keterikatan dan korelasi yang kuat antar variabel lainnya, dimana nilai korelasi antar variabel bernilai lebih dari 0.9 atau mendekati 1.

## Data Preparation

+ Melakukan Pengecekan Type data, Missing Value dan Outlier

    Pengecekan type data dilakukan dengan tujuan agar data yang diolah nantinya tidak menemukan kendala, terdapat beberapa data yang tidak sesuai dengan data typenya maka dilakukan adjusting sesuai dengan data tipenya. Pengecekan pada Missing Value yaitu dengan melakukkan checking semua data jika terdapat data yang tidak memiliki nilai, maka dapat dilakukan pembersihan data tersebut, tetapi karena dataset yang digunakan bersih maka missing value tidak ditemukan Dan untuk mengatasi outlier pada proyek, penulis menggunakan penentuan batas atas dan bawah nilai kuartil pada data dengan menggunakan metode IQR. Hasil dari penangan Outlier data berkurang menjadi 27577.
  
+ Train Test Split

  Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 27577 dibagi menjadi 22061
 untuk data latih dan 5516 untuk data uji.
  
+ Normalization

  Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.StandardScaler.

## Modeling
+ Algoritma
  Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan
  + K-Nearest Neighbour
    K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.

  + Random Forest
    Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `max_depth` = Kedalaman maksimum setiap tree.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

  + Adaboost
    AdaBoost juga disebut Adaptive Boosting adalah teknik dalam machine learning dengan metode ensemble.  Algoritma yang paling umum digunakan dengan AdaBoost adalah pohon keputusan (decision trees) satu tingkat yang berarti memiliki pohon Keputusan dengan hanya 1 split. Pohon-pohon ini juga disebut Decision Stumps. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan cara menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) secara berurutan sehingga membentuk suatu model yang kuat (strong ensemble learner). Proyek ini menggunakan [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

+ Hyperparameter Tuning (Grid Search)
  Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari Grid Search pada proyek ini :
  | model               | best_params                                                      |
  |---------------------|------------------------------------------------------------------|
  | knn                 | {'n_neighbors': 13}                                              |
  | boosting            | {'learning_rate': 0.01, 'n_estimators': 100, 'random_state': 77} |
  | random_forest       | {'max_depth':16, 'n_estimators': 100, 'random_stste': 11}        |


## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE :
<div><img src="https://user-images.githubusercontent.com/107544829/188412654-f5dc0ae1-901b-470e-aae5-1f6b5fb68b4d.png" width="300"/></div>

Berikut hasil evaluasi pada proyek ini :

+ Akurasi
  | model         | accuracy(%)|
  |---------------|------------|
  | KNN           | 62.131671  |
  | RF            | 69.768089  |
  | Boosting      | 56.726024  |

+ Mean Squared Error (MSE)
  <div><img src="https://user-images.githubusercontent.com/107544829/188413846-7d5454b5-7f83-488e-836f-4f3593eb3d5d.png" width="300"/></div>

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini. 
