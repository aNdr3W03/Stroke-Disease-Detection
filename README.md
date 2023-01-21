# Submission 2: Stroke Disease Detection

Nama: Andrew Benedictus Jamesie

Username Dicoding: [andrewbj](https://www.dicoding.com/users/andrewbj)

|     | Deskripsi |
| --- | --------- |
| Dataset | [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| Masalah | Berdasarkan latar belakang di atas, dapat kita ketahui bahwa penyakit stroke adalah masalah kesehatan yang cukup serius, terutama bagi orang yang sudah cukup berumur, memiliki beberapa penyakit penyebab stroke, dan yang paling umum adalah karena memiliki kebiasaan merokok. Meskipun stroke pada umumnya terjadi pada orang yang sudah berumur, namun tidak menutup kemungkinan pula dapat terjadi di usia dewasa menengah atau usia produktif. |
| Solusi Machine Learning | Oleh karena itu, akan lebih baik apabila dilakukan pengecekan secara berkala sebagai tindakan preventif lebih dini terhadap penyakit stroke dengan menggunakan sistem *machine learning* untuk mengetahui kemungkinan risiko terkena penyakit stroke. |
| Metode Pengolahan Data | Metode pengolahan data yang digunakan pada proyek ini adalah dengan menghapus atau *drop* beberapa fitur atau kolom yang tidak dibutuhkan, melakukan tahap *Data Ingestion* dengan membagi *dataset* menjadi *data training* dan *data evaluation* dengan rasio 8:2. Kemudian melakukan tahap *Data Validation* dengan cara melihat statistik data, *data schema*. Setelah itu melakukan tahap *Data Preprocessing* dengan melakukan transformasi fitur input pada data. |
| Arsitektur Model | Arsitektur model yang dibangun menggunakan sebuah *input layer* yang menerima data kategorikal dan numerik yang telah diproses terlebih dahulu. Kemudian terdapat dua buah *hidden layer* (Dense layer) dan sebuah Dropout layer serta sebuah *output layer*. |
| Metrik Evaluasi | Metrik yang digunakan untuk mengevaluasi performa model *machine learning* adalah AUC (*Area Under the ROC Curve*), Precision, Recall, TFMA Example Count, True Positive, True Negatives, False Positive, False Negatives, dan Binary Accuracy. |
| Performa Model | Performa model yang telah dibuat termasuk ke dalam kategori yang cukup baik dan ideal dengan tingkat `binary_accuracy` sebesar 96% dan `val_binary_accuracy` sebesar 95%. Sedangkan untuk nilai `loss` sebesar 0.1510 dan `val_loss` sebesar 0.1580. |
| Opsi Deployment | Proyek *machine learning* Stroke Disease Detection ini telah di-*deploy* menggunakan [Railway App](https://railway.app) sebagai salah satu Platform as a Service (PaaS) yang menyediakan layanan untuk deploying proyek secara gratis. |
| Web App | Model serving dapat diakses melalui [https://stroke-detection.up.railway.app/v1/models/stroke-detection-model/metadata](https://stroke-detection.up.railway.app/v1/models/stroke-detection-model/metadata). |
| Monitoring | Proses *monitoring* pada proyek *machine learning* ini dapat dilakukan menggunakan layanan *open-source*, yaitu [Prometheus](https://prometheus.io). Salah satu proses *monitoring* Prometheus adalah melihat perubahan jumlah permintaan yang dilakukan dengan cara menampilkan status dan informasi *request* beserta jamnya (`:tensorflow:serving:request_count`). |
| Kesimpulan | Model yang telah berhasil dibangun telah diuji coba dapat bekerja dan dapat melakukan klasifikasi apakah seseorang berpotensi untuk terkena penyakit stroke atau tidak dengan tepat. |

## Referensi:

[1] Badan Pusat Statistik, "Banyaknya Desa/Kelurahan Menurut Jenis Bencana Alam dalam Tiga Tahun Terakhir (Desa), 2021", *Badan Pusat Statistik*, 2021, Diambil dari: https://www.bps.go.id/indicator/168/954/1/banyaknya-desa-kelurahan-menurut-jenis-bencana-alam-dalam-tiga-tahun-terakhir.html.

[2] C. M. Annur, "3,59 Juta Orang Terdampak Bencana Alam di Indonesia, Ini Rinciannya", *Katadata.co.id*, 2022, Diambil dari: https://databoks.katadata.co.id/datapublish/2022/10/19/359-juta-orang-terdampak-bencana-alam-di-indonesia-ini-rinciannya.
