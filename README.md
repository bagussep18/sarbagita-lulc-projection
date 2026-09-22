# Proyeksi Perubahan Tutupan Lahan Sarbagita (2020–2026)

Studi kasus GIS Analyst: klasifikasi tutupan lahan Sarbagita (Denpasar,
Badung, Gianyar, Tabanan) tahun 2020 dan 2023, serta proyeksi tahun 2026,
berbasis Python.

## Struktur folder

```
├── README.md
├── requirements.txt
├── training_sample/     (a) sampel training tutupan lahan (.shp)
├── outputs/
│   ├── lc_2020.tif       (b) klasifikasi 2020
│   ├── lc_2023.tif       (c) klasifikasi 2023
│   └── lc_proyeksi_2026.tif  (d) proyeksi 2026
├── accuracy/
│   └── Tabulasi_Uji_Akurasi_Sarbagita.xlsx   (e) OA, Kappa, PA, UA, FoM
└── src/
    ├── 01_classification.py   klasifikasi Random Forest
    └── 02_projection.py       matriks Markov + alokasi spasial CA
```

## Data

- Citra: composite Sentinel-2A 2020 & 2023 (RGB, ~10 m)
- DEM: SRTM 30 m (elevasi & lereng)
- Jaringan jalan dan kawasan hutan (KLHK) — faktor pendorong/pembatas proyeksi
- Training sample: 142 poligon, 6 kelas, digitasi manual dengan acuan citra resolusi tinggi

## Skema kelas

Badan air · Hutan/vegetasi rapat · Kebun/lahan pertanian · Sawah ·
Lahan terbangun · Lahan terbuka

## Metode

1. **Klasifikasi** — Random Forest, fitur RGB + indeks warna + tekstur +
   elevasi/lereng. Data latih/uji dipisah per poligon (bukan per piksel) untuk
   menghindari akurasi yang menggelembung akibat autokorelasi spasial.
2. **Deteksi perubahan** — peta 2023 distabilkan terhadap 2020 memakai
   tingkat keyakinan model (piksel dianggap berubah hanya kalau kedua model
   yakin), untuk menyaring noise klasifikasi yang bisa disalahartikan sebagai
   perubahan lahan.
3. **Matriks Markov** — dihitung dari transisi piksel 2020→2023, dipakai
   untuk memperkirakan luas tiap kelas tahun 2026 (interval waktu sama, 3
   tahun).
4. **Alokasi spasial (Cellular Automata)** — menentukan *lokasi* piksel yang
   paling sesuai untuk tiap transisi, berdasarkan jarak ke jalan, jarak ke
   pusat permukiman, kemiringan lereng, dan ketetanggaan kelas sekitar;
   dibatasi kawasan hutan dan lereng curam.
5. **Validasi** — metode diuji dengan mensimulasikan 2020→2023 dan
   membandingkannya ke peta 2023 aktual (OA, Kappa, dan Figure of Merit).

## Hasil

| Kelas | 2020 (km²) | 2023 (km²) | Proyeksi 2026 (km²) |
|---|---:|---:|---:|
| Badan air | 18,0 | 17,9 | 17,9 |
| Hutan/vegetasi rapat | 485,7 | 479,6 | 477,2 |
| Kebun/lahan pertanian | 364,3 | 366,3 | 370,6 |
| Sawah | 306,3 | 292,7 | 281,9 |
| Lahan terbangun | 440,3 | 439,9 | 442,4 |
| Lahan terbuka | 79,1 | 85,2 | 91,6 |

Tren utama: **sawah terus menyusut** (sebagian ke terbangun dan kebun),
**lahan terbuka bertambah paling besar** secara proporsional, hutan sedikit
menyusut di tepiannya.

Detail lengkap OA, Kappa, Producer's/User's Accuracy, dan Figure of Merit
ada di `accuracy/Tabulasi_Uji_Akurasi_Sarbagita.xlsx`.

## Keterbatasan

- Citra hanya RGB (tanpa NIR/SWIR), sehingga kelas kebun dan lahan terbuka
  relatif lebih sulit dipisahkan dari kelas tetangganya dibanding kelas lain.
- Total perubahan lahan aktual 2020-2023 (~3%) cukup dekat dengan noise dasar
  klasifikasi, sehingga **peta 2026 lebih andal dibaca sebagai proyeksi
  komposisi/tren luas per kelas**, bukan prediksi piksel-demi-piksel yang
  presisi (tercermin dari Figure of Merit yang rendah walau OA/Kappa tinggi).
- Titik pusat permukiman memakai koordinat perkiraan, bukan data survei
  presisi.

## Cara menjalankan ulang

```bash
pip install -r requirements.txt

python src/01_classification.py \
    --image Sarbagita_2020.tif --other-image Sarbagita_2023.tif \
    --training training_sample/Sarbagita_Sample.shp \
    --year 2020 --outdir outputs --dem SRTM_DEM.tif

python src/02_projection.py \
    --lc2020 outputs/lc_2020.tif --lc2023 outputs/lc_2023.tif \
    --dem outputs/dem_features.tif --road dist_jalan.tif \
    --town dist_pusat_permukiman.tif --forest mask_kawasan_hutan.tif \
    --outdir outputs
```

Argumen lengkap ada di docstring masing-masing script.
