# Proyek Pertama Solar Radiation Prediction

#### Disusun oleh : Ichsan Maulana Putra

Ini adalah proyek pertama predictive analytics untuk memenuhi submission dicoding. Proyek ini membangun model machine learning yang dapat memprediksi pancaran sinar radiasi matahari.

## Domain Proyek

### Latar Belakang

Radiasi matahari merupakan salah satu parameter cuaca yang paling berpengaruh dalam sistem iklim, dimana seluruh fenomena cuaca dan iklim pada mulanya disebabkan oleh variasi distribusi penerimaan radiasi matahari. Fluktuasi intensitas radiasi matahari yang diterima di permukaan bumi membentuk pola iklim dalam berbagai skala waktu. Tidak hanya mempengaruhi sistem cuaca dan iklim, pola radiasi matahari juga memberikan informasi penting dalam berbagai sektor, seperti pertanian, sumber daya air, dan energi. Meskipun demikian, radiasi matahari merupakan salah satu parameter cuaca yang masih belum banyak dikaji datanya untuk dikembangkan. Salah satunya pentingnya radiasi matahari disektor energi terhadap pengembangn lokasi yang nantinya digunakan untuk pembangunan PLTS(Pembangkit Listrik Tenaga Surya).

<br>

<div><img src="https://github.com/Icaaan/Solar_Radiation/blob/main/images/0.jpg" width="800"/></div>

<br>

Dalam mencapai hal tersebut, maka dilakukan penelitian untuk memprediksi radiasi matahari menggunakan model machine learning. Diharapkan model ini mampu memprediksi pancaraan radiasi matahari untuk dapat diterapkan diberbagai sektor energi termasuk PLTS(Pembangkit Listrik Tenaga Surya) yang sangat berpengaruh dengan tingkat radiasi matahari. Prediksi ini nantinya dapat dijadikan acuan bagi perusahaan listrik swasta atau skala nasional dalam menentukan loaksi yang cocok untuk diobservasi sebagai lokasi yang dapat menghasilkan energi radiasi matahari yang optimum dimasa yang akan datang.

Referensi : [PREDIKSI RADIASI MATAHARI DENGAN PENERAPAN METODE ELMAN RECURRENT NEURAL NETWORK](https://ejournal.uin-suska.ac.id/index.php/SNTIKI/article/view/7787)

## Business Understanding

### Problem Statements

1. Bagaimana cara melakukan pra-pemrosesan data radiasi matahari sehingga dapat digunakan untuk membuat model yang baik?
2. Fitur apa yang paling berpengaruh terhadap prediksi radiasi matahari?
3. Apakah model machine learning dapat memprediksi radiasi matahari dengan tingkat akurasi yang baik?

### Goals

1. Melakukan pra-pemrosesan data radiasi matahari agar dapat digunakan dalam membangun model.
2. Mengetahui fitur yang paling berpengaruh pada nilai radiasi matahari.
3. Membangun model _machine learning_ untuk memprediksi data harga di masa mendatang dengan tingkat akurasi > 90%.

### Solution Statement

1. Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi outlier.
2. Menyiapkan data agar bisa digunakan dalam membangun model.
3. Melakukan hyperparameter tuning menggunakan grid search dan membangun model regresi yang dapat memprediksi bilangan kontinu. ALgoritma yang dipakai dalam proyek ini adalah K-Nearest Neighbour, Random Forest, dan AdaBoost.

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
  - Dari kolom-kolom tersebut terdapat 4 kolom numerik dengan tipe data float64, yaitu `Radiation, Pressure, WindDirection(Degress), Speed` dan terdapat 2 kolom numerik dengan tipe data int64 yaitu `Temperature, Humidity` yang merupakan fitur numerik. 
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
    
- **Penambahan Fitur**
  <br> diperlukan dibuatnya fitur yang spesifik terhadap waktu Sunset dan Sunrise, karena radiasi matahari berubah sesuai dengan        posisi matahari dan juga durasi radiasi matahari yang diterima bumi. Dengan alasan ini perlu dibuat 2 fitur baru di              dalam dataset sebagai penunjang data radiasi matahari, yaitu : `Duration_time` dan`Rltv_time`  

- **Sebaran atau Distribusi Data pada Setiap Fitur**
  <br> sebelum masuk ke tahap distribusi data, persiapan yang dilakukan yaitu perlu membuat dua variabel baru yaitu variabel 
  <br> Berikut merupakan visualisasi data yang menunjukkan sebaran/distribusi data pada setiap fitur-fitur numerik (`Radiaton, Temperature, pressure, Humidity, WindDirection(Degree), Speed, Duration_time, Rltv_time`) :
  
  - Mengidentifikasi Missing Value dan Outlier
    <br>
    <image src='https://github.com/Icaaan/Solar_Radiation/blob/main/images/1.png' width= 500/>
    <br> Terlihat jika di atas banyak terdapat outlier pada setiap variabel, lalu untuk mengatasinyadengan menerapkan batas bawah dan batas atas menggunakan metode IQR
    
  - Univariate Analysis
    <br>
    <image src='https://github.com/Icaaan/Solar_Radiation/blob/main/images/2.png' width= 500/>
    <br> Terlihat pada grafik bahwa semua data cenderung distribusi nilainya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model nantinya.
    
  - Multivariate Analysis
    <br>
    <image src='https://github.com/Icaaan/Solar_Radiation/blob/main/images/3.png' width= 500/>
    <br> Terlihat bahwa pada grafik kebanyakan bernilai positif karena kebanyakan grafik pada sumbu y dan x mengalami peningkatan yang cukup signifikan membentuk sebuah garis lurus.
    
    <br>
    <image src='https://github.com/Icaaan/Solar_Radiation/blob/main/images/4.png' width= 500/>
    <br> Matrix korelasi mengindikasikan adanya linear korelasi diantara temperature dan solar radiation. Tidak ada data yang berkorelasi linear lain muncul dari feature tersebut.

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
  | knn                 | {'n_neighbors': 10}                                              |
  | boosting            | {'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 33} |
  | random_forest       | {'max_depth':32, 'n_estimators': 75, 'random_stste': 11}        |


## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE :
<div><img src="https://github.com/Icaaan/Solar_Radiation/blob/main/images/8.png" width="300"/></div>

Berikut hasil evaluasi pada proyek ini :

+ Akurasi
  | model         | accuracy(%)|
  |---------------|------------|
  | KNN           | 62.442892  |
  | RF            | 93.793054  |
  | Boosting      | 80.889664  |

+ Mean Squared Error (MSE)
  <div><img src="https://github.com/Icaaan/Solar_Radiation/blob/main/images/5.png" width="300"/></div>
  
Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini. Model ini masih membutuhkan parameter lain sebagai penunjang data agar bisa menghasilkan akurasi yang lebih optimal.
