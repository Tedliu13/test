import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import norm
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from scipy.stats import gamma

# === 設定路徑與參數 ===
#zarr_dir = Path("/media/tedliu/TOSHIBA EXT/ZARR/")
zarr_dir = Path("D:/zarr")
grids_path = "grids.csv"
GWL_file_path = "GWL-models-pr.csv"
variable = "pr"

output_param_dir = Path("parameter")
output_spi3_dir = Path("SPI3Output")
output_monthly_dir = Path("monthlydata")
output_seasonal_dir = Path("SPI3Seasonal")
output_png_dir = Path("SPI3Seasonal/CartopyPNG")

for p in [output_param_dir, output_spi3_dir, output_monthly_dir, output_seasonal_dir, output_png_dir]:
    p.mkdir(parents=True, exist_ok=True)

# === 讀取資料 ===
GWL_df = pd.read_csv(GWL_file_path, encoding='big5')
grids_df = pd.read_csv(grids_path, header=None, names=["lon", "lat"])

# === 定義季節分組 ===
seasons = {
    "spring": [2, 3, 4],
    "meiyu": [5, 6],
    "typhoon": [7, 8, 9],
    "autumn": [10, 11],
    "winter": [12, 1],
    "annual": list(range(1, 13))
}

# === 統計函式 ===
stat_funcs = {
    "mean": "mean",
    "p50": "median",
    "p10": lambda x: np.percentile(x.dropna().values, 10),
    "p25": lambda x: np.percentile(x.dropna().values, 25),
    "p75": lambda x: np.percentile(x.dropna().values, 75),
    "p90": lambda x: np.percentile(x.dropna().values, 90)
}

# === SPI3 計算、月資料與參數儲存 ===
import psutil
import time

proc = psutil.Process(os.getpid())

def print_memory(label):
    mem = proc.memory_info().rss / 1024 / 1024
    print(f"[{label}] 記憶體使用：{mem:.2f} MB")

print("=== SPI3 計算開始 ===")
def fit_gamma_params(data, min_sample=10):
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < min_sample:
        return 1.0, 1.0
    try:
        alpha, loc, beta = gamma.fit(data, floc=0)
        return alpha, beta
    except Exception:
        return 1.0, 1.0

gamma_param_cache = {}
for i, row in GWL_df.iterrows():
    model = row['model']
    if model not in gamma_param_cache:
        gamma_param_cache[model] = {}
    scenario = row['scenario']
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing model: {model}, scenario: {scenario}")

     for phase in ['historical', '1.5°C', '2°C', '3°C', '4°C']:
        if pd.isna(row[phase]):
            continue

        print(f"  Phase: {phase}")
        center_year = int(row[phase])
        start_year = center_year - 9
        end_year = center_year + 10

        if phase == 'historical':
        if (lon, lat) not in gamma_param_cache[model]:
            alphas = []
            betas = []
            acc3 = monthly.rolling(3).sum().dropna()
            for m in range(1, 13):
                sel = acc3[acc3.index.month == m].dropna().values
                alpha, beta = fit_gamma_params(sel)
                alphas.append(alpha)
                betas.append(beta)
            gamma_param_cache[model][(lon, lat)] = (alphas, betas)
            ds = xr.open_zarr(zarr_dir / f"{variable}_QDM_historical_{model}.zarr")
            end_str = f"{end_year}-12-30" if '360_day' in str(ds.time.encoding.get('calendar', '')) else f"{end_year}-12-31"
            ds_sel = ds[variable].sel(time=slice(f"{start_year}-01-01", end_str))
        elif start_year < 2015:
            ds_hist_full = xr.open_zarr(zarr_dir / f"{variable}_QDM_historical_{model}.zarr")
            hist_end_str = "2014-12-30" if '360_day' in str(ds_hist_full.time.encoding.get('calendar', '')) else "2014-12-31"
            ds_hist = ds_hist_full[variable].sel(time=slice(f"{start_year}-01-01", hist_end_str))
            ds_gwl_full = xr.open_zarr(zarr_dir / f"{variable}_QDM_{scenario}_{model}.zarr")
            gwl_end_str = f"{end_year}-12-30" if '360_day' in str(ds_gwl_full.time.encoding.get('calendar', '')) else f"{end_year}-12-31"
            ds_gwl = ds_gwl_full[variable].sel(time=slice("2015-01-01", gwl_end_str))
            ds_sel = xr.concat([ds_hist, ds_gwl], dim="time")
        else:
            ds_gwl_full = xr.open_zarr(zarr_dir / f"{variable}_QDM_{scenario}_{model}.zarr")
            gwl_end_str = f"{end_year}-12-30" if '360_day' in str(ds_gwl_full.time.encoding.get('calendar', '')) else f"{end_year}-12-31"
            ds_sel = ds_gwl_full[variable].sel(time=slice(f"{start_year}-01-01", gwl_end_str))

        if hasattr(ds_sel.time, 'to_index'):
            times = ds_sel.time.to_index()
        else:
            times = pd.to_datetime(ds_sel.time.values)

        df_all = []
        spi_matrix = []
        monthly_matrix = []
        acc3_matrix = []
        time_labels = []

        for month in range(1, 13):
            time_labels.extend([f"{year}_{month:02d}" for year in range(start_year, end_year + 1)])

        if phase == 'historical':
            gamma_param_cache = {}

        for gidx, grid in grids_df.iterrows():
            if gidx % 100 == 0:
                now = time.strftime('%Y-%m-%d %H:%M:%S')
                print_memory(f"[{now}] 開始處理 {model}_{scenario} {gidx+1}/{len(grids_df)}網格，記憶體使用")
            lon, lat = grid["lon"], grid["lat"]
            try:
                series = ds_sel.sel(lat=lat, lon=lon, method="nearest").values
            except:
                series = np.full(len(ds_sel.time), np.nan)

            # 計算月總雨量
            da_series = xr.DataArray(series, coords={"time": times}, dims="time")
            monthly = da_series.resample(time="1ME").sum().to_series()
            monthly_matrix.append(monthly.values)

            # 累積三月
            acc3 = monthly.rolling(3).sum().dropna().reset_index(drop=True)
            acc3_matrix.append(acc3.values)

            # 擬合 gamma 參數（12個月）
            #print(f"  [Grid lon={lon}, lat={lat}] acc3 min={acc3.min():.2f}, max={acc3.max():.2f}")
            if phase == 'historical':
                alphas = []
                betas = []
                acc3 = monthly.rolling(3).sum().dropna()
                for m in range(1, 13):
                    sel = acc3[acc3.index.month == m].dropna().values
                    #print(f"    Month {m:02d}: sample size={len(sel)}, min={sel.min() if len(sel)>0 else 'NA'}, max={sel.max() if len(sel)>0 else 'NA'}")
                    alpha, beta = fit_gamma_params(sel)
                    alphas.append(alpha)
                    betas.append(beta)
                    #print(f"      Fitted alpha={alpha:.3f}, beta={beta:.3f}")
                gamma_param_cache[(lon, lat)] = (alphas, betas)
            else:
                alphas, betas = gamma_param_cache.get((lon, lat), ([1.0]*12, [1.0]*12))

            # 計算 SPI3
            if 1.0 in betas:
                spi = [np.nan] * (len(monthly) - 2)
            else:
                spi = []
                for i in range(2, len(monthly)):
                    acc = monthly.iloc[i-2:i+1].sum()
                    m = monthly.index[i].month
                    a, b = alphas[m - 1], betas[m - 1]
                    spi_val = gamma.cdf(acc, a, scale=b)
                    #if spi_val >= 0.99 or spi_val <= 0.01:
                        #print(f"        Warning: SPI3 CDF={spi_val:.4f} at lon={lon}, lat={lat}, month={m}")
                    spi_val = norm.ppf(spi_val)  # 使用標準常態化

                    spi.append(spi_val)
            spi_matrix.append(spi)

        # 儲存 monthly、acc3、spi3
        month_df = pd.DataFrame(np.array(monthly_matrix), columns=time_labels[:len(monthly_matrix[0])])
        month_df.insert(0, 'lon', grids_df["lon"])
        month_df.insert(1, 'lat', grids_df["lat"])
        month_df.to_csv(output_monthly_dir / f"Monthly_{scenario}_{model}_{phase}.csv", index=False)

        spi_df = pd.DataFrame(np.array(spi_matrix), columns=time_labels[2:2+len(spi_matrix[0])])
        spi_df.insert(0, 'lon', grids_df["lon"])
        spi_df.insert(1, 'lat', grids_df["lat"])
        spi_df.to_csv(output_spi3_dir / f"SPI3_{scenario}_{model}_{phase}.csv", index=False)

        # 輸出為 shapefile
        geometries = [box(lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005) for lon, lat in zip(grids_df["lon"], grids_df["lat"])]
        gdf = gpd.GeoDataFrame(spi_df, geometry=geometries, crs="EPSG:4326")
        gdf.to_file(output_spi3_dir / f"SPI3_{scenario}_{model}_{phase}.shp")

        # 參數儲存
        param_alpha = pd.DataFrame([v[0] for v in gamma_param_cache.values()], columns=[f"Month_{i+1}" for i in range(12)])
        param_beta = pd.DataFrame([v[1] for v in gamma_param_cache.values()], columns=[f"Month_{i+1}" for i in range(12)])
        param_alpha.insert(0, 'lon', [pt[0] for pt in gamma_param_cache.keys()])
        param_alpha.insert(1, 'lat', [pt[1] for pt in gamma_param_cache.keys()])
        param_beta.insert(0, 'lon', [pt[0] for pt in gamma_param_cache.keys()])
        param_beta.insert(1, 'lat', [pt[1] for pt in gamma_param_cache.keys()])
        param_alpha.to_csv(output_param_dir / f"Alpha_{scenario}_{model}_{phase}.csv", index=False)
        param_beta.to_csv(output_param_dir / f"Beta_{scenario}_{model}_{phase}.csv", index=False)

        print_memory(f"[{model}-{scenario}-{phase}] 處理完畢")
        del ds_sel, spi_matrix, monthly_matrix, acc3_matrix
        gc.collect()

# === SPI3 seasonal ensemble 結果整理 ===
from collections import defaultdict

phase_list = ['historical', '1.5°C', '2°C', '3°C', '4°C']
print_memory(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]=== 計算 seasonal ensemble 統計結果 ===")
for phase in ['historical', '1.5°C', '2°C', '3°C', '4°C']:
    print(f"處理 {phase}...")
    all_spi3_files = list(output_spi3_dir.glob(f"SPI3_*_*_{phase}.csv"))
    seasonal_dfs = defaultdict(list)

    for file in all_spi3_files:
        df = pd.read_csv(file)
        df['lon'] = df['lon'].round(2)
        df['lat'] = df['lat'].round(2)
        df.set_index(['lon', 'lat'], inplace=True)

        monthly = df.T  # 轉置後 index 是年月字串
        monthly.index = pd.to_datetime(monthly.index, format="%Y_%m")

        for season, months in seasons.items():
            if season == "winter":
                selected = monthly[(monthly.index.month == 12) | (monthly.index.month == 1)]
            else:
                selected = monthly[monthly.index.month.isin(months)]
            mean_values = selected.mean(axis=0)
            seasonal_dfs[season].append(mean_values)

    result_df = pd.DataFrame(index=seasonal_dfs['annual'][0].index)
    result_df.index.names = ['lon', 'lat']

    for season, dfs in seasonal_dfs.items():
        stack = pd.concat(dfs, axis=1)
        for stat_name, func in stat_funcs.items():
            colname = f"{season}_{stat_name}"
            result_df[colname] = stack.agg(func, axis=1)

    result_df.reset_index(inplace=True)
    result_df.to_csv(output_seasonal_dir / f"Ensemble_SPI3_{phase}.csv", index=False)

# === 輸出 Ensemble shapefile ===
for phase in phase_list:
    df_path = output_seasonal_dir / f"Ensemble_SPI3_{phase}.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
        geometries = [box(lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005) for lon, lat in zip(df["lon"], df["lat"])]
        rename_dict = {col: col[:10] for col in df.columns if len(col) > 10}
        gdf = gpd.GeoDataFrame(df.rename(columns=rename_dict), geometry=geometries, crs="EPSG:4326")
        gdf.to_file(output_seasonal_dir / f"Ensemble_SPI3_{phase}.shp")

# === Cartopy 圖層設定 ===
season_list = list(seasons.keys())
phase_list = ['historical', '1.5°C', '2°C', '3°C', '4°C']
cmap = plt.get_cmap("RdBu").reversed()
norm = mcolors.BoundaryNorm(boundaries=[x * 0.25 for x in range(-16, 17)], ncolors=cmap.N)

# === SPI3 cartopy 畫圖函式 ===
def plot_spi3_cartopy(df, phase, season):
    col = f"{season}_mean"
    if col not in df.columns:
        return

    geometries = [
        box(lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005)
        for lon, lat in zip(df["lon"], df["lat"])
    ]
    gdf = gpd.GeoDataFrame(df[[col]].copy(), geometry=geometries, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(9, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([119.5, 122.5, 21.8, 25.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax, edgecolor='none', transform=ccrs.PlateCarree(), legend=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("SPI3")

    ax.set_title(f"{phase} - {season} SPI3 Mean", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_png_dir / f"{phase}_{season}_cartopy.png", dpi=300)
    plt.close()

# === SPI3 quick 分布檢查 ===
print_memory(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]=== SPI3 quick check histogram ===")
for phase in phase_list:
    spi_file = output_seasonal_dir / f"Ensemble_SPI3_{phase}.csv"
    if spi_file.exists():
        df = pd.read_csv(spi_file)
        for season in ['spring', 'meiyu', 'typhoon', 'autumn', 'winter', 'annual']:
            col = f"{season}_mean"
            if col in df.columns:
                plt.figure(figsize=(8, 4))
                df[col].hist(bins=30, color='skyblue', edgecolor='black')
                plt.title(f"SPI3 {phase} - {season} distribution")
                plt.xlabel('SPI3')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_png_dir / f"{phase}_{season}_hist.png", dpi=200)
                plt.close()

# === 批次繪圖 ===
if __name__ == "__main__":
    for phase in phase_list:
        df_path = output_seasonal_dir / f"Ensemble_SPI3_{phase}.csv"
        if df_path.exists():
            df = pd.read_csv(df_path)
            for season in season_list:
                plot_spi3_cartopy(df, phase, season)
print_memory(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]======== 計算完畢 ========")
