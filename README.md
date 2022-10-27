# Proyek Pertama _Solar Radiation Prediction_

#### Disusun oleh : Ichsan Maulana Putra

Ini adalah proyek pertama _predictive analytics_ untuk memenuhi _submission_ dicoding. Proyek ini membangun _model machine learning_ yang dapat memprediksi pancaran sinar radiasi matahari.

## Domain Proyek

### Latar Belakang

Radiasi matahari merupakan salah satu parameter cuaca yang paling berpengaruh dalam sistem iklim, dimana seluruh fenomena cuaca dan iklim pada mulanya disebabkan oleh variasi distribusi penerimaan radiasi matahari. Fluktuasi intensitas radiasi matahari yang diterima di permukaan bumi membentuk pola iklim dalam berbagai skala waktu. Tidak hanya mempengaruhi sistem cuaca dan iklim, pola radiasi matahari juga memberikan informasi penting dalam berbagai sektor, seperti pertanian, sumber daya air, dan energi. Meskipun demikian, radiasi matahari merupakan salah satu parameter cuaca yang masih belum banyak dikaji datanya untuk dikembangkan [1]. Salah satunya pentingnya radiasi matahari disektor energi terhadap pengembangn lokasi yang nantinya digunakan untuk pembangunan PLTS(Pembangkit Listrik Tenaga Surya).


 ![0](https://user-images.githubusercontent.com/48026319/198274865-07a86e25-c949-44ca-9de2-69ae6f92bd29.jpg)

 Gambar 1. _Solar Photovoltaic Plant_


Dalam mencapai hal tersebut, maka dilakukan penelitian untuk memprediksi radiasi matahari menggunakan model machine learning. Diharapkan model ini mampu memprediksi pancaraan radiasi matahari untuk dapat diterapkan diberbagai sektor energi termasuk PLTS(Pembangkit Listrik Tenaga Surya) yang sangat berpengaruh dengan tingkat radiasi matahari. Prediksi ini nantinya dapat dijadikan acuan bagi perusahaan listrik swasta atau skala nasional dalam menentukan loaksi yang cocok untuk diobservasi sebagai lokasi yang dapat menghasilkan energi radiasi matahari yang optimum dimasa yang akan datang.


## Business Understanding

### Problem Statements

1. Bagaimana cara melakukan pra-pemrosesan data radiasi matahari sehingga dapat digunakan untuk membuat model yang baik?
2. Fitur apa yang paling berpengaruh terhadap prediksi radiasi matahari?
3. Apakah _model machine learning_ dapat memprediksi radiasi matahari dengan tingkat akurasi yang baik?

### Goals

1. Melakukan pra-pemrosesan data radiasi matahari agar dapat digunakan dalam membangun model.
2. Mengetahui fitur yang paling berpengaruh pada nilai radiasi matahari.
3. Membangun model _machine learning_ untuk memprediksi data harga di masa mendatang dengan tingkat akurasi > 90%.

### Solution Statement

1. Menganalisis data dengan melakukan _univariate analysis_ dan _multivariate analysis_. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi _outlier_.
2. Menyiapkan data agar bisa digunakan dalam membangun model.
3. Melakukan _hyperparameter tuning menggunakan grid search_ dan membangun model regresi yang dapat memprediksi bilangan kontinu. ALgoritma yang dipakai dalam proyek ini adalah _K-Nearest Neighbour_, _Random Forest_, dan _AdaBoost_.

## Data Understanding
- **Informasi Dataset**
  <br> Dataset yang digunakan pada proyek ini yaitu dataset lengkap dengan perngukuran radiasi matahari selama 4 bulan, informasi lebih lanjut mengenai _dataset_ tersebut dapat lihat pada tabel berikut:
  
  Tabel 1. Informasi Dataset
  | Jenis                   | Keterangan                                                                                         |
  | ----------------------- | -------------------------------------------------------------------------------------------------- |
  | Sumber                  | Dataset: [Kaggle](https://www.kaggle.com/datasets/dronio/SolarEnergy?select=SolarPrediction.csv)   |
  | Dataset Owner           | ANDREY                                                                                             |
  | Lisensi                 | https://opendatacommons.org/licenses/dbcl/1-0/                                                     |
  | Kategori                | _SolarRadiation, Energy_                                                                           |
  | Usability               | 8.24                                                                                               |
  | Jenis dan Ukuran Berkas | CSV (2.9 MB)                                                                                       |

  Setelah melakukan observasi pada dataset yang diunduh melalui _link_ Kaggle yaitu `SolarPrediction.csv', didapatkan informasi sebagai berikut :
  
  - Terdapat 32686 baris (_records_ atau jumlah pengamatan) yang berisi informasi mengenai data pengkuran.
  - Terdapat 11 kolom yaitu `UNIXTime, Data, Time, Radiation, Temperature, _Pressure, Humidity, WindDirection(Degress), Speed, TimeSunRise, TimeSunSet` yang merupakan variabel - variabel pada data
  - Dari kolom-kolom tersebut terdapat 4 kolom numerik dengan tipe data float64, yaitu `Radiation, Pressure, WindDirection(Degress), Speed` dan terdapat 2 kolom numerik dengan tipe data int64 yaitu `Temperature, Humidit_` yang merupakan fitur numerik. 
  - Terdapat 2 kolom dengan tipe datetime yaitu `UNIXTime, Data, Time, TimeSunRise, TimeSunSet`
  - Tidak terdapat _missing value_ pada dataset. 
  
  Untuk penjelasan mengenai variabel-variabel pada dataset dapat dilihat pada poin-poin berikut ini:

    * _UNIXTime_ = adalah jumlah detik yang telah berlalu sejak 00:00:00 UTC pada 1 Januari 1970 [s]
    * _data,Time_ = Tanggal/waktu saat pengambilan data [%Y-%m-%d %H:%M:%S]
    * _Radiation_ = radiasi yang dipancarkan oleh matahari [W/m^2]
    * _Temperatue_ = Temperature saat pengukuran terjadi [F]
    * _Atmospheric pressure_ = Tekanan atmosfer bumi [Hg]
    * _Humidity_ = Kelembapan saat pengukuran terjadi [%]
    * _Wind speed_ = Kecepatan Angin Pada Saat Pengukuran Terjadi[miles/h]
    * _Wind direction_ = Arah angin yang dilambangkan dengan [degrees]
    * _Time SunRise/Sunset_ = Waktu Matahari terbit dan terbenam [HST(Hawai time)]
    
- **Penambahan Fitur**
  <br>
  Diperlukan dibuatnya fitur yang spesifik terhadap waktu Sunset dan Sunrise, karena radiasi matahari berubah sesuai dengan posisi matahari dan juga durasi               radiasi matahari yang diterima bumi. Dengan alasan ini perlu dibuat 2 fitur baru di dalam dataset sebagai penunjang data radiasi matahari, yaitu : `Duration_time`     dan `Rltv_time`  

- **Sebaran atau Distribusi Data pada Setiap Fitur**
  <br>
  Sebelum masuk ke tahap distribusi data, persiapan yang dilakukan yaitu perlu membuat dua variabel baru yaitu variabel 
  Berikut merupakan visualisasi data yang menunjukkan sebaran/distribusi data pada setiap fitur-fitur numerik (`Radiaton, Temperature, pressure, _umidity, WindDirection(Degree), Speed, Duration_time, Rltv_time`) :
  
    - Mengidentifikasi _Missing Value_ dan _Outlier_
     
    ![1](https://user-images.githubusercontent.com/48026319/198277559-18690669-13a8-4a24-a6e6-0ff9fa0de540.png)
     
    Gambar 2. Data _Outlier_ radiasi matahari
      
    Terlihat jika di atas banyak terdapat outlier pada setiap variabel, lalu untuk mengatasinyadengan menerapkan batas bawah dan batas atas menggunakan metode IQR
    
    - _Univariate Analysis_
     
    ![2](https://user-images.githubusercontent.com/48026319/198277672-eceafb85-242d-4b81-8c66-43307b20a23e.png)
    
    Gambar 3. Data _Univariate Analysis_ radiasi matahari
    
    Terlihat pada grafik bahwa semua data cenderung distribusi nilainya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model nantinya.
    
    - _Multivariate Analysis_
    
    ![3](https://user-images.githubusercontent.com/48026319/198277736-b7f525cd-0512-4cb1-bacc-5e570747632b.png)
    
    Gambar 4. Data _Multivariate Analysis_ radiasi matahari
    
    Terlihat bahwa pada grafik kebanyakan bernilai positif karena kebanyakan grafik pada sumbu y dan x mengalami peningkatan yang cukup signifikan membentuk sebuah         garis lurus.
    
    -  _Matrix Correlation_ 
    
    ![4](https://user-images.githubusercontent.com/48026319/198278231-5f633912-9a68-412d-83b0-a12a37192897.png)
      
    Gambar 5. Data _Matrix Correlation_ radiasi matahari
    
    _Matrix Correlation_ mengindikasikan adanya _linear_ korelasi diantara _temperature_ dan _solar radiation_. Tidak ada data yang berkorelasi _linear_ lain muncul        dari _feature_ tersebut.

## Data Preparation

+ Melakukan Pengecekan _Data Type, Missing Value_ dan _Outlier_

    Pengecekan tipe data dilakukan dengan tujuan agar data yang diolah nantinya tidak menemukan kendala, terdapat beberapa data yang tidak sesuai dengan data tipenya maka dilakukan _adjusting_ sesuai dengan data tipenya. Pengecekan pada _Missing Value_ yaitu dengan melakukkan _checking_ semua data, jika terdapat data yang tidak memiliki nilai, maka dapat dilakukan pembersihan data tersebut. Tetapi karena dataset yang digunakan bersih maka _missing value_ tidak ditemukan Dan untuk mengatasi _outlier_ pada proyek, penulis menggunakan penentuan batas atas dan bawah nilai kuartil pada data dengan menggunakan metode IQR. Hasil dari penangan _Outlier_ data berkurang menjadi 27577.
  
+ _Train Test Split_

  _Train test split_ aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 27577 dibagi menjadi 22061 untuk data latih dan 5516 untuk data uji.
  
+ _Normalization_

  Algoritma _machine learning_ akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan _sklearn.preprocessing.StandardScaler._[2]

## Modeling
+ Algoritma
  <br>Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu _K-Nearest Neighbour, Random Forest_, dan Adaboost
  + _K-Nearest Neighbour_
    _K-Nearest Neighbour_ bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan _sklearn.neighbors.KNeighborsRegressor_[3] dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.

  + _Random Forest_
    <br>Algoritma _random forest_ adalah teknik dalam machine learning dengan metode _ensemble_. Teknik ini beroperasi dengan membangun banyak _decision tree_ pada waktu pelatihan. Proyek ini menggunakan _sklearn.ensemble.RandomForestRegressor_[4] dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `max_depth` = Kedalaman maksimum setiap tree.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

  + Adaboost
    <br>AdaBoost juga disebut _Adaptive Boosting_ adalah teknik dalam _machine learning_ dengan metode _ensemble_.  Algoritma yang paling umum digunakan dengan AdaBoost adalah pohon keputusan (_decision trees_) satu tingkat yang berarti memiliki pohon Keputusan dengan hanya 1 _split_. Pohon-pohon ini juga disebut _Decision Stumps_. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan cara menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) secara berurutan sehingga membentuk suatu model yang kuat (_strong ensemble learner_). Proyek ini menggunakan _sklearn.ensemble.AdaBoostRegressor_[5] dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

+ _Hyperparameter Tuning (Grid Search)_
  <br>
  _Hyperparameter tuning_ adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari _Grid Search_ pada proyek ini :

  Tabel 2. Hasil _Hyperparameter Tuning (Grid Search)_
  | model               | best_params                                                      |
  |---------------------|------------------------------------------------------------------|
  | knn                 | {'n_neighbors': 10}                                              |
  | boosting            | {'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 33} |
  | random_forest       | {'max_depth':32, 'n_estimators': 75, 'random_stste': 11}        |


## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan _mean squared error (MSE)_. Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). _Mean squared error_ (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE :

![8](https://user-images.githubusercontent.com/48026319/198298023-2f85a22a-a66c-496a-97c5-e30e18b87337.png)

Gambar 6. Formula _Mean squared error_

Berikut hasil evaluasi pada proyek ini :

+ _Accuracy Model_

  Tabel 3. _Accuracy Model_
  | model         | accuracy(%)|
  |---------------|------------|
  | KNN           | 62.442892  |
  | RF            | 93.793054  |
  | Boosting      | 80.889664  |

+ _Mean Squared Error (MSE)_

  ![5](https://user-images.githubusercontent.com/48026319/198277076-51c88739-2285-4d09-ace4-b3c677d15b4f.png)
  
  Gambar 7. Data hasil Keluaran _Mean Squared Error (MSE)_

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini. Model ini masih membutuhkan parameter lain sebagai penunjang data agar bisa menghasilkan akurasi yang lebih optimal.

## Referensi

[1] https://megasains.gawbkt.id/index.php/megasains/article/view/45

[2] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

[3] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

[4] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

[5] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
