"""
Alokasi spasial CA-Markov untuk proyeksi tutupan lahan Sarbagita.

Prinsip (mengikuti pendekatan CA-Markov standar, mis. di TerrSet/MOLUSCE):
  - Markov memberi KUANTITAS (berapa piksel tiap kelas perlu bertambah/berkurang)
  - Suitability + CA memberi LOKASI (piksel mana yang paling cocok menerima kelas baru)
Kuota per kelas TIDAK dipaksa mengikuti pasangan asal-tujuan Markov secara rinci;
komposisi asal kelas yang terkonversi murni muncul dari kesesuaian spasial piksel.
Ini konsisten dengan cara kerja CA-Markov konvensional.

Faktor:
  Pendorong : jarak ke jalan, jarak ke pusat permukiman, kemiringan lereng (datar),
              elevasi (utk terbangun), efek ketetanggaan (neighborhood, radius 5x5)
  Pembatas  : kawasan hutan (mask biner, dari KLHK) -> larang ekspansi kebun/sawah/
              terbangun/terbuka ke dalamnya; lereng > SLOPE_LIMIT_DEG -> larang
              ekspansi terbangun & terbuka

Kelas donor (bisa kehilangan area): Hutan, Kebun, Sawah, Lahan terbuka
Kelas statis (tidak diutak-atik)   : Badan air, dan Lahan terbangun sebagai donor
                                     (praktis tidak pernah berkurang di data historis)
Kelas penerima (bisa menambah area), diproses berurutan sesuai prioritas driver:
  1. Lahan terbangun (paling deterministik: jalan+pusat kota)
  2. Lahan terbuka   (mirip terbangun, transisi/pra-pembangunan)
  3. Kebun           (didorong ketetanggaan, lebih difus)

Validasi: simulasikan 2020->2023 dengan metode yang SAMA (kuota dari transisi historis
2020->2023 yang sudah distabilkan), lalu bandingkan dengan peta 2023 aktual: Overall
Accuracy, Kappa, dan Figure of Merit (Pontius et al.) yang secara khusus menilai
ketepatan LOKASI piksel yang berubah, bukan cuma kecocokan keseluruhan peta.
"""
import argparse
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CLASS_NAMES = {1: "Badan air", 2: "Hutan/vegetasi rapat", 3: "Kebun/lahan pertanian",
              4: "Sawah", 5: "Lahan terbangun", 6: "Lahan terbuka"}
DONOR_CLASSES = [2, 3, 4, 6]                    # kelas yang boleh kehilangan piksel
RECEIVER_ORDER = [5, 6, 3]                      # urutan prioritas kelas penerima
NEI_SIZE = 5                                    # jendela efek ketetanggaan (piksel, ~50 m)
SLOPE_LIMIT_DEG = 35                            # lereng di atas ini: tak boleh jadi terbangun/terbuka
ROAD_SAT_M = 1500                               # jarak jalan dianggap "jauh" mulai dari sini
TOWN_SAT_M = 10000                              # jarak pusat permukiman dianggap "jauh" mulai dari sini
ELEV_SAT_M = 500                                # elevasi dianggap "tinggi" mulai dari sini (utk terbangun)


def load_factors(dem_path, road_path, town_path, forest_path):
    with rasterio.open(dem_path) as s:
        elev, slope = s.read(1), s.read(2)
    with rasterio.open(road_path) as s:
        dist_road = s.read(1)
    with rasterio.open(town_path) as s:
        dist_town = s.read(1)
    with rasterio.open(forest_path) as s:
        forest = s.read(1)
    return elev, slope, dist_road, dist_town, forest


def neighborhood_fraction(state, klass, size=NEI_SIZE):
    return uniform_filter((state == klass).astype("float32"), size=size, mode="nearest")


def build_suitability(target, state, elev, slope, dist_road, dist_town, forest, valid):
    """Suitability [0,1] untuk kelas `target`, dari faktor + ketetanggaan pada `state`."""
    road_n = 1 - np.clip(dist_road / ROAD_SAT_M, 0, 1)
    town_n = 1 - np.clip(dist_town / TOWN_SAT_M, 0, 1)
    flat_n = 1 - np.clip(slope / 30.0, 0, 1)
    elev_n = 1 - np.clip(elev / ELEV_SAT_M, 0, 1)

    if target == 5:      # Lahan terbangun
        neigh = neighborhood_fraction(state, 5)
        S = 0.35 * road_n + 0.25 * town_n + 0.20 * flat_n + 0.10 * elev_n + 0.10 * neigh
        S[slope > SLOPE_LIMIT_DEG] = 0
        S[forest == 1] = 0   # kawasan hutan: larang ekspansi terbangun ke dalamnya
    elif target == 6:    # Lahan terbuka (transisi, mengikuti pola mirip terbangun)
        neigh = neighborhood_fraction(state, 6)
        S = 0.40 * road_n + 0.30 * town_n + 0.30 * neigh
        S[slope > SLOPE_LIMIT_DEG] = 0
        S[forest == 1] = 0
    elif target == 3:    # Kebun (didorong ketetanggaan kebun & tepi hutan)
        neigh_k = neighborhood_fraction(state, 3)
        neigh_h = neighborhood_fraction(state, 2)
        S = 0.55 * neigh_k + 0.25 * neigh_h + 0.20 * (1 - town_n)
        S[forest == 1] = 0   # kebun tidak boleh "resmi" meluas ke kawasan hutan
    elif target == 4:    # Sawah (butuh datar, dekat sawah eksisting)
        neigh = neighborhood_fraction(state, 4)
        S = 0.55 * flat_n + 0.45 * neigh
        S[slope > SLOPE_LIMIT_DEG] = 0
        S[forest == 1] = 0
    elif target == 2:    # Hutan (regrowth/suksesi): didorong tepi hutan eksisting & lereng curam,
                         # dan JUSTRU diperbolehkan (didorong) di dalam kawasan hutan legal
        neigh = neighborhood_fraction(state, 2)
        steep_n = np.clip(slope / 30.0, 0, 1)
        S = 0.45 * neigh + 0.25 * steep_n + 0.30 * forest.astype("float32")
    else:
        raise ValueError(target)

    S[~valid] = 0
    return np.clip(S, 0, 1).astype("float32")


def allocate_pairwise(state, pair_quota_px, elev, slope, dist_road, dist_town, forest, valid,
                      receivers=(1, 2, 3, 4, 5, 6)):
    """
    Alokasi CA berbasis matriks transisi PENUH (per pasangan asal->tujuan), bukan cuma
    demand bersih per kelas. pair_quota_px: {(from_id, to_id): jumlah piksel}.
    Pasangan diproses dari kuota terbesar ke terkecil; sekali piksel dipindah, tidak
    dipindah lagi pada pasangan lain di ronde yang sama.
    """
    out = state.copy()
    already_moved = np.zeros(state.shape, dtype=bool)
    realized = {}
    # cache suitability per kelas tujuan (dihitung dari state AWAL, satu langkah non-iteratif)
    suit_cache = {}
    for target in receivers:
        if target in (1,):   # badan air: tidak dialokasikan sebagai penerima
            continue
        suit_cache[target] = build_suitability(target, state, elev, slope, dist_road, dist_town, forest, valid)

    for (frm, to), n_target in sorted(pair_quota_px.items(), key=lambda kv: -kv[1]):
        n_target = int(round(n_target))
        if n_target <= 0 or frm == to or to not in suit_cache or frm == 1:
            realized[(frm, to)] = 0
            continue
        S = suit_cache[to]
        eligible = valid & ~already_moved & (out == frm) & (S > 0)
        idx = np.flatnonzero(eligible)
        if idx.size == 0:
            realized[(frm, to)] = 0
            continue
        scores = S.ravel()[idx]
        k = min(n_target, idx.size)
        top = idx[np.argpartition(-scores, k - 1)[:k]] if k < idx.size else idx
        flat_out = out.ravel(); flat_out[top] = to
        already_moved.ravel()[top] = True
        realized[(frm, to)] = k
    return out, realized


def figure_of_merit(ref_t0, ref_t1, sim_t1, valid):
    """
    Figure of Merit (Pontius et al. 2004) per definisi standar:
    FoM = hits / (hits + misses + false_alarms), dihitung pada domain PERUBAHAN saja
    (piksel yang berubah di referensi ATAU di simulasi).
    hits          : berubah di keduanya, DAN ke kelas yang sama
    misses        : berubah di referensi, tapi simulasi bilang tidak berubah / kelas beda
    false_alarms  : berubah di simulasi, tapi referensi bilang tidak berubah / kelas beda
    """
    ref_change = valid & (ref_t0 != ref_t1)
    sim_change = valid & (ref_t0 != sim_t1)
    hits = int((ref_change & sim_change & (ref_t1 == sim_t1)).sum())
    misses = int((ref_change & ~(sim_change & (ref_t1 == sim_t1))).sum())
    false_alarms = int((sim_change & ~(ref_change & (ref_t1 == sim_t1))).sum())
    denom = hits + misses + false_alarms
    return hits / denom if denom else float("nan"), hits, misses, false_alarms


def evaluate(ref_t0, ref_t1, sim_t1, valid, label):
    m = valid & (ref_t1 > 0) & (sim_t1 > 0)
    labels = list(CLASS_NAMES)
    cm = confusion_matrix(ref_t1[m], sim_t1[m], labels=labels)
    oa = np.trace(cm) / cm.sum()
    kappa = cohen_kappa_score(ref_t1[m], sim_t1[m], labels=labels)
    fom, hits, misses, fa = figure_of_merit(ref_t0, ref_t1, sim_t1, valid)
    print(f"\n[{label}] OA={oa:.4f}  Kappa={kappa:.4f}  FoM={fom:.4f}  "
         f"(hits={hits:,} misses={misses:,} false_alarms={fa:,})")
    cm_df = pd.DataFrame(cm, index=[f"Aktual: {CLASS_NAMES[c]}" for c in labels],
                         columns=[f"Simulasi: {CLASS_NAMES[c]}" for c in labels])
    print(cm_df.to_string())
    return {"label": label, "OA": oa, "Kappa": kappa, "FoM": fom,
            "hits": hits, "misses": misses, "false_alarms": fa}, cm_df


def px_area_km2(transform, lat0):
    dx = transform.a * 111320 * np.cos(np.radians(lat0))
    dy = transform.a * 110540
    return abs(dx * dy) / 1e6


def demand_from_matrix(comp_from, P, names=CLASS_NAMES):
    """comp_from: Series luas km2 per kelas pada t0. P: matriks probabilitas transisi (baris=asal)."""
    comp_to = pd.Series(comp_from.values @ P.values, index=comp_from.index)
    return comp_to


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="run_v5_DEM/dem_features.tif")
    ap.add_argument("--road", default="data/dist_jalan.tif")
    ap.add_argument("--town", default="data/dist_pusat_permukiman.tif")
    ap.add_argument("--forest", default="data/mask_kawasan_hutan.tif")
    ap.add_argument("--lc2020", default="run_v5_DEM/lc_2020.tif")
    ap.add_argument("--lc2023", default="run_v5_DEM/lc_2023_stable.tif")
    ap.add_argument("--matrix", default="markov_probabilitas_transisi_stabil.csv")
    ap.add_argument("--outdir", default="proj_out")
    a = ap.parse_args()

    import os
    os.makedirs(a.outdir, exist_ok=True)

    elev, slope, dist_road, dist_town, forest = load_factors(a.dem, a.road, a.town, a.forest)
    with rasterio.open(a.lc2020) as s:
        lc2020 = s.read(1); profile = s.profile.copy(); transform = s.transform
        lat0 = (s.bounds.top + s.bounds.bottom) / 2
    with rasterio.open(a.lc2023) as s:
        lc2023 = s.read(1)
    valid = (lc2020 > 0) & (lc2023 > 0)
    px_km2 = px_area_km2(transform, lat0)

    # Matriks probabilitas transisi dihitung ulang di sini (bukan dari CSV) agar konsisten
    # dengan CLASS_NAMES/label di atas, dan selalu sinkron dengan raster lc_2020/lc_2023 yang dipakai.
    cm_px = confusion_matrix(lc2020[valid], lc2023[valid], labels=list(CLASS_NAMES))
    cm_px = pd.DataFrame(cm_px, index=list(CLASS_NAMES.values()), columns=list(CLASS_NAMES.values()))
    P = cm_px.div(cm_px.sum(axis=1), axis=0)
    id_of = {v: k for k, v in CLASS_NAMES.items()}

    # ---------- VALIDASI: simulasikan 2020 -> 2023 dgn kuota PER PASANGAN asal->tujuan ----------
    # Kuota diambil langsung dari jumlah piksel aktual yang bertransisi 2020->2023 (gross,
    # bukan cuma net), supaya churn dua arah (mis. hutan->kebun DAN kebun->hutan) sama-sama
    # dialokasikan alih-alih saling meniadakan seperti pada pendekatan net-demand.
    pair_quota_val = {(id_of[a], id_of[b]): cm_px.loc[a, b]
                      for a in CLASS_NAMES.values() for b in CLASS_NAMES.values() if a != b}
    print("10 pasangan transisi terbesar (piksel) yang dipakai sbg kuota validasi:")
    for (f, t), n in sorted(pair_quota_val.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {CLASS_NAMES[f]:20s} -> {CLASS_NAMES[t]:20s} {n:8d} px ({n*px_km2:6.2f} km2)")

    sim2023, realized = allocate_pairwise(lc2020, pair_quota_val, elev, slope, dist_road, dist_town, forest, valid)
    tot_target = sum(pair_quota_val.values()); tot_real = sum(realized.values())
    print(f"\nTotal kuota diminta: {tot_target*px_km2:.1f} km2 | total realisasi: {tot_real*px_km2:.1f} km2")

    metrics, cm_df = evaluate(lc2020, lc2023, sim2023, valid, "Validasi: simulasi 2023 vs aktual")
    pd.DataFrame([metrics]).to_csv(f"{a.outdir}/validasi_metrics.csv", index=False)
    cm_df.to_csv(f"{a.outdir}/validasi_confusion_matrix.csv")
    with rasterio.open(f"{a.outdir}/lc_2023_simulasi.tif", "w", **profile) as d:
        d.write(sim2023, 1)

    # ---------- PROYEKSI 2026: kuota per-pasangan dari P (asumsi pola transisi berulang) ----------
    comp2023 = pd.Series({n: (lc2023[valid] == id_of[n]).sum() * px_km2 for n in CLASS_NAMES.values()})
    pair_quota_26 = {}
    for a_name in CLASS_NAMES.values():
        for b_name in CLASS_NAMES.values():
            if a_name == b_name:
                continue
            km2 = comp2023[a_name] * P.loc[a_name, b_name]
            pair_quota_26[(id_of[a_name], id_of[b_name])] = km2 / px_km2

    proj2026, realized26 = allocate_pairwise(lc2023, pair_quota_26, elev, slope, dist_road, dist_town, forest, valid)
    tot_target26 = sum(pair_quota_26.values()); tot_real26 = sum(realized26.values())
    print(f"\n[Proyeksi 2026] Total kuota: {tot_target26*px_km2:.1f} km2 | realisasi: {tot_real26*px_km2:.1f} km2")

    with rasterio.open(f"{a.outdir}/lc_proyeksi_2026.tif", "w", **profile) as d:
        d.write(proj2026, 1)

    out = pd.DataFrame({"2023_km2": comp2023, "proyeksi_2026_km2":
                        pd.Series({n: (proj2026[valid] == id_of[n]).sum() * px_km2 for n in CLASS_NAMES.values()})})
    out["delta_km2"] = out["proyeksi_2026_km2"] - out["2023_km2"]
    out.round(2).to_csv(f"{a.outdir}/komposisi_2026.csv")
    print("\nKomposisi akhir 2026 (dari peta hasil alokasi):")
    print(out.round(1).to_string())
    print(f"\nSelesai. Output di: {a.outdir}/")
