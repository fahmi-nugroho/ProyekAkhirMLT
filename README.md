# Laporan Proyek Machine Learning - Fahmi Nugroho Alibasyah

## Project Overview
Untuk proyek rekomendasi ini saya memilih sistem rekomendasi bertemakan pariwisata di Indonesia, saya memilih tema tersebut karena melihat angka kasus covid-19 di Indonesia yang semakin menurun membuat PPKM (Pemberlakuan Pembatasan Kegiatan Masyarakat) yang diterapkan pemerintah semakin dilonggarkan, ini menjadi kabar yang baik pada dunia pariwisata yang selama ini menjadi salah satu sektor yang paling terdampak covid-19. Dengan dilonggarkannya PPKM oleh pemerintah masyarakat akhirnya bisa melakukan perjalanan, tentunya dengan menerapkan protokol kesehatan yang ada. Dengan alasan itu semua maka menurut saya masyarakat akan sangat terbantu ketika ingin melakukan liburan jika ada suatu aplikasi yang bisa merekomendasikan tempat-tempat yang bagus untuk didatangi.

## Business Understanding
* Pernyataan masalah
    * Bagaimana menentukan tempat wisata yang sekiranya disukai oleh seseorang.
* Tujuan
    * Membuat model machine learning yang dapat merekomendasikan tempat-tempat wisata yang sesuai dengan preferensi seseorang.
* Pernyataan solusi
    * Membuat sistem rekomendasi menggunakan algoritma Content Based Filtering
    * Membuat sistem rekomendasi menggunakan algoritma Collaborative Filtering.

## Data Understanding
Data yang saya gunakan merupakan data Indonesia Tourism Destination yang saya ambil pada website [Kaggle](https://www.kaggle.com/aprabowo/indonesia-tourism-destination?select=tourism_with_id.csv). Data tersebut berisi beberapa file csv seperti :
* tourism_with_id.csv yang berisi beberapa fitur seperti:
    * Place_Id (ID dari tempat wisata)
    * Place_Name (Nama dari tempat wisata)
    * Description (Deskripsi dari tempat wisata)
    * Category (Kategori tempat wisata)
    * City (Letak kota dari tempat wisata)
    * Price (Kisaran harga masuk tempat wisata)
    * Rating (Peringkat dari tempat wisata)
* user.csv yang berisi beberapa fitur seperti:
    * User_Id (ID dari user)
    * Location (Lokasi user tinggal)
    * Age (Umur user)
* tourism_rating.csv yang berisi beberapa fitur seperti:
    * User_Id (ID dari user)
    * Place_Id (ID dari tempat wisata)
    * Place_Ratings (Peringkat dari user)

File csv dan fitur-fitur diatas akan digunakan untuk membuat model sistem rekomendasi yang akan kita buat nanti.

Data kategori tempat wisata

![Visualisasi kategori](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/hist1.png?raw=true)

Data lokasi tempat wisata

![Visualisasi kategori](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/hist2.png?raw=true)

Data rating tempat wisata

![Visualisasi rating](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/hist3rev.png?raw=true)

## Data Preparation
* Menggabungkan data tourism_with_id.csv dan tourism_rating.csv berdasarkan fitur Place_Id
Penggabungan data dilakukan agar kita bisa mengetahui berapa rating yang diberikan oleh masing-masing orang kepada tempat wisata tertentu, misalnya user 1 memberikan rating sekian kepada tempat wisata A.
    ``` python
    tourisms_info = pd.merge(ratings, tourisms , on='Place_Id', how='left')
    ```
* Menyeleksi data yang akan digunakan
Tidak semua fitur yang ada pada file csv akan digunakan, karena pada file csv ada beberapa data yang tidak lengkap atau tidak akan digunakan dalam membuat model. Pada kasus ini saya menghapus kolom 'Time_Minutes', 'Coordinate', 'Lat', 'Long', dan index yang undifined. Lalu mengecek apakah ada data yang kosong pada data yang sudah kita bersihkan, pada kasus ini data yang sudah saya bersihkan tidak mengandung nilai null, jadi tidak perlu proses drop lagi. Setelah semua data terbebas dari nilai null maka data yang duplikat juga akan di hilangkan.
    ``` python
    # Kode untuk menghapus beberapa kolom yang tidak digunakan
    tourisms_info.drop(['Time_Minutes', 'Coordinate', 'Lat', 'Long', tourisms_info.columns[-1], tourisms_info.columns[-2]], axis='columns', inplace=True)
    # Kode untuk memeriksa apakah ada nilai null atau tidak
    tourisms_info.isnull().sum()
    # Kode untuk menghilangkan data duplikat
    preparation = fix_tourisms.drop_duplicates('Place_Id')
    ```
* Mengonversi data series Place_Id, Place_Name, dan Category menjadi list dan menjadikanya satu dataframe
    ``` python
    # Mengonversi data series ‘Place_Id’ menjadi dalam bentuk list
    place_id = preparation['Place_Id'].tolist()
     
    # Mengonversi data series ‘Place_Name’ menjadi dalam bentuk list
    place_name = preparation['Place_Name'].tolist()
     
    # Mengonversi data series ‘Category’ menjadi dalam bentuk list
    place_category = preparation['Category'].tolist()
     
    tourisms_new = pd.DataFrame({
        'id': place_id,
        'name': place_name,
        'category': place_category
    })
    ```
* Melakukan encoding untuk User_Id
    ``` python
    # Mengubah User_Id menjadi list tanpa nilai yang sama
    user_ids = df['User_Id'].unique().tolist()
    print('list User_Id: ', user_ids)
     
    # Melakukan encoding User_Id
    user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
    print('encoded User_Id : ', user_to_user_encoded)
     
    # Melakukan proses encoding angka ke ke User_Id
    user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
    print('encoded angka ke User_Id: ', user_encoded_to_user)
    ```
* Melakukan encoding untuk Place_Id
    ``` python
    # Mengubah Place_Id menjadi list tanpa nilai yang sama
    tourisms_ids = df['Place_Id'].unique().tolist()
     
    # Melakukan proses encoding Place_Id
    tourisms_to_tourisms_encoded = {x: i for i, x in enumerate(tourisms_ids)}
     
    # Melakukan proses encoding angka ke Place_Id
    tourisms_encoded_to_tourisms = {i: x for i, x in enumerate(tourisms_ids)}
    ```
* Melakukan pembagian data untuk training dan validasi
Dengan melakukan pembagian dataset kita dapat menilai bagaimana performa yang dihasilkan model kita ketika bertemu data-data yang belum pernah dilihat pada proses latihan sebelumnya.
    ``` python
    # Mengacak dataset
    df = df.sample(frac=1, random_state=42)
    # Membuat variabel x untuk mencocokkan data user dan resto menjadi satu value
    x = df[['user', 'tourisms']].values
     
    # Membuat variabel y untuk membuat rating dari hasil 
    y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
     
    # Membagi menjadi 80% data train dan 20% data validasi
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    ```

# Modeling
Saya menggunakan dua algoritma untuk membuat sistem rekomendasi tempat wisata di indonesia.
* Content Based Filtering.
Pada teknik ini produk/item atribut yang dijadikan sebagai referensi kesamaan untuk hasil rekomendasi, model tidak memperhatikan informasi dari user lainnya. Pada model yang saya gunakan item yang digunakan adalah kategori tempat wisata. Lalu jika sudah ditentukan item yang akan digunakan, kita menggunakan TF-IDF untuk menemukan representasi fitur penting dari setiap kategori tempat wisata. Selanjutnya menghitung derajat kesamaan (similarity degree) antar tempat wisata menggunakan Cosine Similarity, jika sudah ditemukan tempat wisata mana yang mirip, maka rekomendasi akan dilakukan.
    
    Metode berbasis konten tampaknya jauh lebih sulit saat awal (cold start) jika dibandingkan dengan pendekatan kolaboratif, karena pengguna atau item baru hanya dapat dijelaskan oleh karakteristiknya, yaitu konten dan saran yang relevan dapat dilakukan untuk entitas baru ini. Hanya pengguna baru atau item dengan fitur yang sebelumnya tidak terlihat yang akan mengalami kelemahan ini, tetapi setelah sistem cukup terlatih, kemungkinan ini kecil atau bahkan tidak ada sama sekali.
    
* Collaborative Filtering.
    Pada collaborative filtering attribut yang digunakan bukan konten tetapi user behaviour. contohnya kita merekomendasikan suatu item berdasarkan dari riwayat rating dari user tersebut maupun user lain. Untuk melakukan rekomendasi dapat menggunakan item-based maupun user-based. Pada item based kita berangkat dari item sebagai bari dan user sebagai kolom, kemudian hitung similaritynya setalah mendapat nilai similarit kita sorting descending, nah disini kita bisa improvisasi dengan melakukan filter pada film-film yang sudah pernah ditonton. lalu kita dapat memilih Top N rekomendasi.

    Satu-satunya masalah dengan metode ini adalah bahwa prediksi model untuk pengguna tertentu, pasangan item adalah produk titik dari penyematan yang sesuai. Jadi, jika suatu item tidak terlihat selama pelatihan, sistem umumnya tidak dapat membuat embedding untuk itu dan karenanya tidak dapat mengkueri model dengan item ini. Masalah ini dikenal sebagai masalah (Cold Start).

## Evaluation
Untuk menilai performa model saya menggunakan dua metrik evaluasi. Saya memilih dua metrik dibawah karena saya memnggunakan dua algoritma yang berbeda, sehingga diperlukan dua metrik evaluasi yang berbeda pula.
* Precission
Precission adalah kemampuan pengklasifikasi untuk tidak melabeli instance positif yang sebenarnya negatif. Untuk setiap kelas, itu didefinisikan sebagai rasio positif benar dengan jumlah positif benar dan positif palsu.
![Precission](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/eval1.png?raw=true)
* Root Mean Square Error (RMSE)
Pengertian Root Mean Square Error (RMSE) adalah  metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil
![RMSE](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/eval2.png?raw=true)

Penerapan dalam kode:
``` python
# menghitung precission
TP = len(hasil[hasil.category == kategori_rekomendasi])
FP = len(hasil[hasil.category != kategori_rekomendasi])

Precision = TP / (TP+FP)
# menggunakan RMSE untuk model yang saya buat
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

Hasil Evaluasi Content Based Filtering

![Evaluasi Content Based Filtering](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/hasil1.png?raw=true)

Hasil Evaluasi Collaborative Filtering

![Evaluasi Collaborative Filtering](https://github.com/fahmi-nugroho/ProyekAkhirMLT/blob/main/hasil2.png?raw=true)

# Referensi
* Aprilia Saptu Ningrum, Heru Cahya Rustamaji, dan Yuli Fauziah, "Content Based dan Collavorative Filtering Pada Rekomendasi Tujuan Pariwisata di Daerah Yogyakarta", TELEMATIKA, Vol. 16, No.1, 2019, doi: [10.31315/telematika.v16i1.3023](https://doi.org/10.31315/telematika.v16i1.3023).
* Mufidatul Islamiyah, Puji Subekti, dan Titania Dwi Andini, "Pemanfaatan Metode Item Based Collaborative Filtering Untuk Rekomendasi Wisata Di Kabupaten Malang", Jurnal Ilmiah Teknologi Informasi Asia Vol.13, No. 2, 2019, doi: [10.32815/jitika.v13i2.70](https://doi.org/10.32815/jitika.v13i2.70).
