"""
Klasifikasi tutupan lahan Sarbagita (Random Forest) dari composite Sentinel-2A
berupa citra RGB 8-bit (band: vis-red, vis-green, vis-blue; nodata = 0).

Citra hanya berisi R, G, B (tanpa NIR/SWIR), sehingga fitur yang dipakai:
  R, G, B, koordinat kromatik (r, g), Excess Green (ExG), NGRDI, kecerahan,
  saturasi, dan tekstur lokal (simpangan baku kecerahan, jendela 5x5 dan 11x11).

Alur:
  1. Baca sampel poligon; pilih poligon yang berlaku untuk tahun yang diproses
     (kolom thn_valid: 'both', '2020', '2023'; kosong dianggap 'both')
  2. Baca citra per blok (dengan padding agar tekstur tidak terputus di batas blok)
  3. Mask awan: piksel sangat terang DAN jauh lebih terang daripada citra tahun lain
     (--other-image). Permukaan terang permanen (atap putih, pasir, kapur) tidak
     dianggap awan. Tanpa --other-image, dipakai ambang kecerahan saja.
  4. Ambil piksel poligon, batasi maksimum piksel per poligon (--max-px-poly)
  5. Validasi: (a) hold-out per poligon 70/30, (b) validasi silang per poligon
     (StratifiedGroupKFold). Metrik: OA, Kappa, PA, UA, confusion matrix
  6. Model akhir dilatih pada seluruh sampel, lalu memprediksi seluruh citra

Contoh:
  python 01_classification.py --image Sarbagita_2020.tif --other-image Sarbagita_2023.tif \
      --training Sarbagita_Sample.shp --year 2020 --outdir outputs
  python 01_classification.py --image Sarbagita_2023.tif --other-image Sarbagita_2020.tif \
      --training Sarbagita_Sample.shp --year 2023 --outdir outputs [--match-to Sarbagita_2020.tif]

Kolom shapefile: class_id (wajib, 1-6); poly_id atau id (ID unik);
                 thn_valid (opsional: both / 2020 / 2023).
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import binary_dilation, uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
SEED = 42
TILE = 1024            # ukuran blok pemrosesan (piksel)
PAD = 6                # padding per blok, >= setengah jendela tekstur terbesar
TRAIN_FRACTION = 0.7   # porsi poligon untuk pelatihan pada hold-out, per kelas
NODATA = 0             # nilai nodata citra (di luar batas Sarbagita)
CLOUD_MIN = 170        # awan: min(R,G,B) di atas nilai ini ...
CLOUD_DIFF = 60        # ... dan lebih terang sekian level dari citra tahun lain
CLOUD_BUFFER = 2       # perluasan mask awan (piksel), menutup tepi awan
TEXTURE_SIZES = (5, 11)

CLASS_NAMES = {
    1: "Badan air",
    2: "Hutan/vegetasi rapat",
    3: "Kebun/lahan pertanian",
    4: "Sawah",
    5: "Lahan terbangun",
    6: "Lahan terbuka",
}

FEATURE_NAMES = ["R", "G", "B", "r_chrom", "g_chrom", "ExG", "NGRDI",
                 "brightness", "saturation"] + [f"tex_std_{s}" for s in TEXTURE_SIZES]
DEM_FEATURE_NAMES = ["elevation_m", "slope_deg"]


# ----------------------------------------------------------------------------
# FITUR
# ----------------------------------------------------------------------------
def local_std(x, valid, size):
    """Simpangan baku lokal yang mengabaikan piksel nodata (normalized convolution)."""
    w = valid.astype("float32")
    n = uniform_filter(w, size, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        m = uniform_filter(x * w, size, mode="constant") / n
        m2 = uniform_filter(x * x * w, size, mode="constant") / n
    out = np.sqrt(np.clip(m2 - m * m, 0, None))
    out[~np.isfinite(out)] = 0
    return out.astype("float32")


def build_features(arr, valid, dem_arr=None):
    """arr: (3, h, w) -> (n_fitur, h, w) float32. dem_arr opsional: (2, h, w) = elevasi, lereng,
    pada grid yang PERSIS sama dengan citra (dihasilkan oleh prepare_dem_features)."""
    a = arr.astype("float32")
    R, G, B = a
    s = R + G + B
    with np.errstate(divide="ignore", invalid="ignore"):
        r, g, b = R / s, G / s, B / s
        exg = 2 * g - r - b
        ngrdi = (G - R) / (G + R)
        sat = (a.max(axis=0) - a.min(axis=0)) / a.max(axis=0)
    bright = s / 3
    feats = [R, G, B, r, g, exg, ngrdi, bright, sat]
    for size in TEXTURE_SIZES:
        feats.append(local_std(bright, valid, size))
    if dem_arr is not None:
        feats.extend([dem_arr[0], dem_arr[1]])
    f = np.nan_to_num(np.stack(feats).astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    f[:, ~valid] = 0
    return f


def iter_windows(height, width, size=TILE):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def prepare_window(src, win, lut=None, other=None, dem=None):
    """
    Baca satu blok (dengan padding), hitung fitur, lalu potong ke ukuran blok.
    dem: dataset rasterio 2-band (elevasi, lereng) pada grid yang SAMA PERSIS dengan
    src (dibuat oleh prepare_dem_features), sehingga window yang sama berlaku untuk keduanya.
    Return: feats (n, h, w), usable (valid & bukan awan), cloud (h, w) bool.
    """
    pwin = Window(win.col_off - PAD, win.row_off - PAD,
                  win.width + 2 * PAD, win.height + 2 * PAD)
    raw = src.read(window=pwin, boundless=True, fill_value=NODATA)
    valid = (raw != NODATA).any(axis=0)
    minv = raw.min(axis=0).astype("int16")
    cloud = valid & (minv > CLOUD_MIN)
    if other is not None:
        omin = other.read(window=pwin, boundless=True, fill_value=NODATA).min(axis=0).astype("int16")
        cloud &= (minv - omin) > CLOUD_DIFF
    if CLOUD_BUFFER > 0 and cloud.any():
        cloud = binary_dilation(cloud, iterations=CLOUD_BUFFER) & valid
    arr = raw
    if lut is not None:
        arr = np.stack([lut[i][raw[i]] for i in range(3)])
        arr[:, ~valid] = NODATA
    dem_arr = dem.read(window=pwin, boundless=True, fill_value=0).astype("float32") if dem is not None else None
    feats = build_features(arr, valid, dem_arr)
    sl = (slice(None), slice(PAD, -PAD), slice(PAD, -PAD))
    valid_c, cloud_c = valid[sl[1:]], cloud[sl[1:]]
    return feats[sl], valid_c & ~cloud_c, cloud_c


def prepare_dem_features(dem_path, ref_path, out_path):
    """
    Samakan DEM ke grid citra acuan (transform/CRS/ukuran identik) dan hitung lereng (derajat).
    Simpan sebagai GeoTIFF 2-band (elevasi, lereng) di out_path. Hanya perlu dijalankan sekali
    per pasangan (DEM, area); citra 2020 dan 2023 di sini berbagi grid yang sama.
    """
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        dst_transform, dst_crs, dst_shape = ref.transform, ref.crs, (ref.height, ref.width)
        lat0 = (ref.bounds.top + ref.bounds.bottom) / 2

    with rasterio.open(dem_path) as src:
        elev = np.zeros(dst_shape, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=elev,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)

    res_deg = dst_transform.a
    dx = res_deg * 111320 * np.cos(np.radians(lat0))
    dy = res_deg * 110540
    gy, gx = np.gradient(elev, dy, dx)
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2))).astype("float32")

    profile.update(count=2, dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(elev, 1); dst.write(slope, 2)
        dst.set_band_description(1, "elevation_m")
        dst.set_band_description(2, "slope_deg")
    return out_path


# ----------------------------------------------------------------------------
# HISTOGRAM MATCHING (opsional, menyamakan radiometrik antar tahun)
# ----------------------------------------------------------------------------
def band_histograms(src):
    hist = np.zeros((3, 256), dtype="float64")
    for win in iter_windows(src.height, src.width):
        raw = src.read(window=win)
        valid = (raw != NODATA).any(axis=0)
        m = valid & ~(raw.min(axis=0) > CLOUD_MIN)
        for i in range(3):
            hist[i] += np.bincount(raw[i][m], minlength=256)
    return hist


def make_luts(hist_src, hist_ref):
    """LUT per band: memetakan distribusi citra sumber ke distribusi citra acuan."""
    luts, levels = [], np.arange(256)
    for i in range(3):
        cs = np.cumsum(hist_src[i]) / hist_src[i].sum()
        cr = np.cumsum(hist_ref[i]) / hist_ref[i].sum()
        luts.append(np.clip(np.round(np.interp(cs, cr, levels)), 0, 255).astype("uint8"))
    return luts


# ----------------------------------------------------------------------------
# SAMPEL
# ----------------------------------------------------------------------------
def load_training(path, crs):
    gdf = gpd.read_file(path).to_crs(crs)
    if "class_id" not in gdf.columns:
        raise ValueError("Shapefile training harus punya kolom 'class_id'.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    if "poly_id" not in gdf.columns:
        gdf["poly_id"] = gdf["id"] if "id" in gdf.columns else np.arange(1, len(gdf) + 1)
    gdf["poly_id"] = gdf["poly_id"].astype(int)
    gdf["class_id"] = gdf["class_id"].astype(int)
    if gdf["poly_id"].duplicated().any():
        raise ValueError("poly_id/id harus unik.")
    if "thn_valid" not in gdf.columns:
        gdf["thn_valid"] = "both"
    tv = gdf["thn_valid"].fillna("").astype(str).str.strip().str.lower()
    tv = tv.str.replace(r"\.0$", "", regex=True)   # 2020.0 -> 2020 bila tersimpan sebagai angka
    gdf["thn_valid"] = np.where(tv.isin(["", "nan", "none", "null"]), "both", tv)
    return gdf


def filter_year(gdf, year):
    return gdf[gdf["thn_valid"].isin(["both", str(year)])].reset_index(drop=True)


def extract_samples(src, gdf, lut=None, other=None, dem=None):
    """Ekstrak piksel di dalam poligon. Return X, y, poly (id poligon)."""
    shapes = [(g, int(p)) for g, p in zip(gdf.geometry, gdf["poly_id"])]
    poly_to_class = dict(zip(gdf["poly_id"], gdf["class_id"]))
    Xs, ys, ps = [], [], []
    for win in iter_windows(src.height, src.width):
        ids = rasterize(shapes, out_shape=(win.height, win.width),
                        transform=src.window_transform(win), fill=0, dtype="int32")
        if not ids.any():
            continue
        feats, usable, _ = prepare_window(src, win, lut, other, dem)
        m = (ids > 0) & usable
        if not m.any():
            continue
        Xs.append(feats[:, m].T)
        p = ids[m]
        ps.append(p)
        ys.append(np.array([poly_to_class[i] for i in p], dtype="int16"))
    if not Xs:
        raise RuntimeError("Tidak ada piksel training yang beririsan dengan citra.")
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ps)


def cap_pixels(X, y, poly, max_px, seed=SEED):
    """Batasi jumlah piksel per poligon (acak, seed tetap) agar poligon besar tidak mendominasi."""
    if not max_px:
        return X, y, poly
    rng = np.random.default_rng(seed)
    keep = []
    for pid in np.unique(poly):
        idx = np.where(poly == pid)[0]
        if len(idx) > max_px:
            idx = rng.choice(idx, max_px, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], poly[keep]


def split_by_polygon(gdf, frac=TRAIN_FRACTION, seed=SEED):
    """Bagi poligon (bukan piksel) latih/uji, terstratifikasi per kelas.
    Dihitung dari SEMUA poligon agar pembagian sama untuk 2020 dan 2023."""
    rng = np.random.default_rng(seed)
    train_ids = set()
    for _, grp in gdf.groupby("class_id"):
        ids = grp["poly_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_train = max(1, int(round(len(ids) * frac)))
        if len(ids) > 1:
            n_train = min(n_train, len(ids) - 1)
        train_ids.update(ids[:n_train].tolist())
    return train_ids


# ----------------------------------------------------------------------------
# VALIDASI
# ----------------------------------------------------------------------------
def accuracy_report(y_true, y_pred, labels=None, names=None):
    """
    Confusion matrix, Overall Accuracy, Cohen's Kappa, Producer's dan User's
    Accuracy per kelas. Baris = referensi, kolom = prediksi.
    Return: (cm_df, per_class_df, summary_df)
    """
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    names = names or CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    diag = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = np.where(cm.sum(axis=1) > 0, diag / cm.sum(axis=1), np.nan)  # producer
        ua = np.where(cm.sum(axis=0) > 0, diag / cm.sum(axis=0), np.nan)  # user
    oa = diag.sum() / cm.sum()
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    lab_names = [names.get(l, str(l)) for l in labels]

    cm_df = pd.DataFrame(cm, index=[f"Ref: {n}" for n in lab_names],
                         columns=[f"Pred: {n}" for n in lab_names])
    per_class = pd.DataFrame({
        "class_id": labels, "class_name": lab_names,
        "n_reference": cm.sum(axis=1), "n_predicted": cm.sum(axis=0),
        "producer_accuracy": pa, "user_accuracy": ua,
    })
    summary = pd.DataFrame({
        "metric": ["overall_accuracy", "cohen_kappa", "n_samples"],
        "value": [oa, kappa, int(cm.sum())],
    })
    return cm_df, per_class, summary


def grouped_cv_predict(X, y, poly, n_splits, n_estimators):
    """Prediksi out-of-fold dengan validasi silang per poligon (tanpa kebocoran spasial)."""
    n_min = pd.Series(y).groupby(y).apply(lambda s: len(np.unique(poly[s.index]))).min()
    n_splits = int(max(2, min(n_splits, n_min)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = np.zeros_like(y)
    for tr, te in sgkf.split(X, y, groups=poly):
        m = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred, n_splits


def save_results(results, outdir, year):
    """results: {tag: (cm_df, per_class, summary)} -> CSV per tag + satu Excel."""
    outdir = Path(outdir)
    for tag, (cm_df, per_class, summary) in results.items():
        cm_df.to_csv(outdir / f"accuracy_{year}_{tag}_confusion_matrix.csv")
        per_class.to_csv(outdir / f"accuracy_{year}_{tag}_per_class.csv", index=False)
        summary.to_csv(outdir / f"accuracy_{year}_{tag}_summary.csv", index=False)
    try:
        with pd.ExcelWriter(outdir / f"accuracy_{year}.xlsx") as xw:
            for tag, (cm_df, per_class, summary) in results.items():
                summary.to_excel(xw, sheet_name=f"summary_{tag}", index=False)
                per_class.to_excel(xw, sheet_name=f"per_class_{tag}", index=False)
                cm_df.to_excel(xw, sheet_name=f"confusion_{tag}")
    except ImportError:
        print("openpyxl tidak terpasang; hanya CSV yang disimpan.")


# ----------------------------------------------------------------------------
# PREDIKSI
# ----------------------------------------------------------------------------
def predict_raster(src, model, out_lc, out_cloud, lut=None, other=None, dem=None, out_conf=None):
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    conf_profile = src.profile.copy()
    conf_profile.update(count=1, dtype="float32", nodata=0, compress="lzw")
    n_valid = n_cloud = 0
    dst_cf = rasterio.open(out_conf, "w", **conf_profile) if out_conf else None
    with rasterio.open(out_lc, "w", **profile) as dst_lc, \
         rasterio.open(out_cloud, "w", **profile) as dst_cl:
        for win in iter_windows(src.height, src.width):
            feats, usable, cloud = prepare_window(src, win, lut, other, dem)
            out = np.zeros((win.height, win.width), dtype="uint8")
            conf = np.zeros((win.height, win.width), dtype="float32")
            if usable.any():
                proba = model.predict_proba(feats[:, usable].T)
                out[usable] = model.classes_[np.argmax(proba, axis=1)].astype("uint8")
                conf[usable] = proba.max(axis=1).astype("float32")
            dst_lc.write(out, 1, window=win)
            dst_cl.write(cloud.astype("uint8"), 1, window=win)
            if dst_cf is not None:
                dst_cf.write(conf, 1, window=win)
            n_valid += int(usable.sum() + cloud.sum())
            n_cloud += int(cloud.sum())
    if dst_cf is not None:
        dst_cf.close()
    return n_cloud / max(n_valid, 1)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def run(image, training, year, outdir, other_image=None, match_to=None, dem=None,
        n_estimators=300, max_px_poly=100, cv_folds=5, predict=True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    other = rasterio.open(other_image) if other_image else None

    with rasterio.open(image) as src:
        if src.count != 3:
            raise ValueError(f"Script ini untuk citra 3 band (RGB); citra punya {src.count} band.")

        lut = None
        if match_to:
            with rasterio.open(match_to) as ref:
                lut = make_luts(band_histograms(src), band_histograms(ref))
            print(f"Histogram matching aktif: {Path(image).name} -> {Path(match_to).name}")

        dem_ds = None
        feature_names = FEATURE_NAMES
        if dem:
            dem_path = outdir / "dem_features.tif"
            if not dem_path.exists():
                prepare_dem_features(dem, image, dem_path)
            dem_ds = rasterio.open(dem_path)
            feature_names = FEATURE_NAMES + DEM_FEATURE_NAMES
            print(f"Fitur DEM aktif: {dem_path.name} ({DEM_FEATURE_NAMES})")

        gdf_all = load_training(training, src.crs)
        train_ids = split_by_polygon(gdf_all)          # sama untuk semua tahun
        gdf = filter_year(gdf_all, year)
        print(f"Poligon berlaku untuk {year}: {len(gdf)} dari {len(gdf_all)}")

        X, y, poly = extract_samples(src, gdf, lut, other, dem_ds)

        # Ringkasan sampel per poligon (untuk diagnosa)
        raw_cnt = pd.Series(poly).value_counts()
        summ = gdf[["poly_id", "class_id", "thn_valid"]].copy()
        summ["n_px_usable"] = summ["poly_id"].map(raw_cnt).fillna(0).astype(int)
        summ["n_px_used"] = summ["n_px_usable"].clip(upper=max_px_poly or None)
        summ.to_csv(outdir / f"sample_summary_{year}.csv", index=False)
        dropped = summ[summ["n_px_usable"] == 0]
        if len(dropped):
            print("Poligon tanpa piksel valid (nodata/awan), dilewati:",
                  dict(zip(dropped["poly_id"], dropped["class_id"])))

        X, y, poly = cap_pixels(X, y, poly, max_px_poly)
        print(f"Piksel training (setelah batas {max_px_poly}/poligon): {len(y)}")
        print(pd.Series(y).map(CLASS_NAMES).value_counts().to_string())

        results = {}
        # (a) hold-out per poligon
        is_train = np.isin(poly, list(train_ids))
        m = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED).fit(X[is_train], y[is_train])
        results["holdout"] = accuracy_report(y[~is_train], m.predict(X[~is_train]), labels=list(CLASS_NAMES))
        # (b) validasi silang per poligon
        pred_cv, k = grouped_cv_predict(X, y, poly, cv_folds, n_estimators)
        results["cv"] = accuracy_report(y, pred_cv, labels=list(CLASS_NAMES))
        save_results(results, outdir, year)
        for tag, (_, per_class, summary) in results.items():
            print(f"\n[{tag}{'' if tag == 'holdout' else f' {k}-fold'}]")
            print(summary.to_string(index=False))
            print(per_class.round(3).to_string(index=False))

        if not predict:
            return
        # Model akhir: seluruh sampel
        model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=SEED, oob_score=True).fit(X, y)
        print(f"\nModel akhir OOB (piksel, cenderung optimistis): {model.oob_score_:.4f}")
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}) \
            .sort_values("importance", ascending=False) \
            .to_csv(outdir / f"feature_importance_{year}.csv", index=False)

        cloud_frac = predict_raster(src, model, outdir / f"lc_{year}.tif",
                                    outdir / f"cloud_mask_{year}.tif", lut, other, dem_ds,
                                    out_conf=outdir / f"confidence_{year}.tif")
        print(f"Tersimpan: {outdir / f'lc_{year}.tif'} (piksel awan = 0, {100*cloud_frac:.2f}% area valid)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--year", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--other-image", default=None,
                    help="citra tahun lain (untuk mask awan temporal)")
    ap.add_argument("--match-to", default=None,
                    help="citra acuan histogram matching (mis. 2020 saat memproses 2023)")
    ap.add_argument("--dem", default=None,
                    help="raster DEM (mis. SRTM); disamakan otomatis ke grid --image, "
                         "menambah fitur elevasi dan lereng")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-px-poly", type=int, default=100,
                    help="maks piksel per poligon (0 = tanpa batas)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--no-predict", action="store_true",
                    help="hanya validasi, tanpa memprediksi seluruh citra")
    a = ap.parse_args()
    run(a.image, a.training, a.year, a.outdir, a.other_image, a.match_to, a.dem,
        a.n_estimators, a.max_px_poly, a.cv_folds, predict=not a.no_predict)
