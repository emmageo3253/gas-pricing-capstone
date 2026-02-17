# =========================
# GasPricesSpain.py  (FULL REWRITE + COMPETITION + OVERWRITE FIX)
# Precioil station pull + daily Gasolina95 panel
# + Overpass (OSM) tourism/business features with robust caching
# + Competition proxy:
#     (1) local (2km) variance/std of weekly neighbor mean prices (station-week)
#     (2) teacher metric: avg of EACH neighbor's weekly price-change frequency (station-week)
# + ALSO writes station-level AVERAGES across weeks (easy to inspect in Excel)
#
# IMPORTANT OUTPUT BEHAVIOR:
# - Outputs are written with an overwrite-safe function that writes to a temp file then os.replace()
# - If Excel/OneDrive locks the target file, the script raises a clear error instead of silently writing alternates
# =========================

import os
import json
import time
import random
import re
import math
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_URL = "https://api.precioil.es"

FECHA_INICIO = "2010-01-01"
FECHA_FIN    = "2030-12-31"

G95_CODE = 6

# Throttle
SLEEP_SECONDS = 0.35   # Precioil
BASE_SLEEP    = 3.0    # Overpass

# Caches
CACHE_DIR = "cache_hist"
OVERPASS_CACHE_DIR = "cache_overpass"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OVERPASS_CACHE_DIR, exist_ok=True)

# Overpass endpoints
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Outputs
OUT_PANEL_CSV      = "panel_daily_gasolina95.csv"
OUT_OSM_CSV        = "overpass_station_features.csv"

# Competition output (station-week panel)
OUT_COMP_CSV       = "weekly_local_price_variance_2km.csv"
# Competition averages output (station-level averages across weeks)
OUT_COMP_AVG_CSV   = "competition_station_averages_2km.csv"

# Overpass controls
RADIUS_M   = 2000        # OSM feature radius
TARGET_N   = 200         # stations per run (if sampling)
RUN_SAMPLE = True        # True = sample remaining; False = process all remaining

# Competition controls
COMP_RADIUS_M = 2000
SAMPLE_N_STATIONS = 50   # sample size for competition measure (stations i)
MIN_NEIGHBORS_WITH_PRICE = 2  # need >=2 neighbor prices to compute variance/std

# Anchors
anchors = [
    ("Madrid", 40.4168, -3.7038),
]

# =========================
# IO HELPERS (OVERWRITE FIX)
# =========================
def write_csv_overwrite(df: pd.DataFrame, path: str):
    """
    Overwrite CSV reliably:
      1) write to temp
      2) replace target atomically-ish via os.replace()

    If Excel/OneDrive locks the target, this will raise PermissionError
    (so you KNOW it didn't overwrite).
    """
    if df is None or df.empty:
        raise ValueError(f"Refusing to write empty DataFrame to {path}")

    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)

    try:
        os.replace(tmp, path)  # overwrites if exists
        print("Overwrote:", os.path.abspath(path), "| rows:", len(df))
    except PermissionError as e:
        # Clean temp file so it doesn't confuse you
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise PermissionError(
            f"Could not overwrite {path} because it appears locked (Excel/OneDrive). "
            f"Close the file and run again."
        ) from e


# =========================
# PRECIOIL HELPERS
# =========================
def get_stations_in_radius(lat, lon, radio_km=15, limite=50, max_paginas=20):
    """Collect unique stations from /estaciones/radio with pagination."""
    seen = set()
    stations = []

    for pagina in range(1, max_paginas + 1):
        params = {
            "latitud": lat,
            "longitud": lon,
            "radio": radio_km,
            "pagina": pagina,
            "limite": limite,
        }
        r = requests.get(f"{BASE_URL}/estaciones/radio", params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        new = 0
        for s in batch:
            sid = s.get("idEstacion")
            if sid is not None and sid not in seen:
                seen.add(sid)
                stations.append(s)
                new += 1

        if new == 0:
            break

        time.sleep(SLEEP_SECONDS)

    return stations


def fetch_history_cached(id_estacion, fecha_inicio, fecha_fin):
    """Fetch /estaciones/historico/{id} with caching + throttling."""
    cache_path = os.path.join(CACHE_DIR, f"hist_{id_estacion}_{fecha_inicio}_{fecha_fin}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    params = {"fechaInicio": fecha_inicio, "fechaFin": fecha_fin}
    r = requests.get(f"{BASE_URL}/estaciones/historico/{id_estacion}", params=params, timeout=30)
    time.sleep(SLEEP_SECONDS)
    r.raise_for_status()
    j = r.json()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)

    return j


def history_to_daily_g95(hist_json):
    """Convert one station's history JSON into daily Gasolina95 panel."""
    if not (isinstance(hist_json, dict) and "data" in hist_json):
        return pd.DataFrame()

    df = pd.DataFrame(hist_json["data"])
    if df.empty:
        return df

    df["precio"] = pd.to_numeric(df.get("precio"), errors="coerce")
    df["idFuelType"] = pd.to_numeric(df.get("idFuelType"), errors="coerce")
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")

    df = df[df["idFuelType"] == G95_CODE].copy()
    if df.empty:
        return df

    df["date"] = df["timestamp"].dt.date

    daily = (
        df.groupby(["idEstacion", "date"], as_index=False)
          .agg(
              precio_mean=("precio", "mean"),
              precio_last=("precio", "last"),
              n_obs=("precio", "size")
          )
    )
    return daily


# =========================
# COMPETITION HELPERS
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat,lon) points in km."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# =========================
# OVERPASS HELPERS
# =========================
def _cache_path_for_station(sid, radius_m):
    return os.path.join(OVERPASS_CACHE_DIR, f"osm_{sid}_{radius_m}m.json")


def overpass_post_with_retries(query, timeout=120, max_tries=6):
    """POST an Overpass query with retries + exponential backoff + mirror rotation."""
    start = random.randrange(len(OVERPASS_URLS))
    last_err = None

    for attempt in range(max_tries):
        url = OVERPASS_URLS[(start + attempt) % len(OVERPASS_URLS)]
        try:
            r = requests.post(url, data=query, timeout=timeout)

            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"Transient HTTP {r.status_code} from {url}")

            r.raise_for_status()

            ctype = (r.headers.get("Content-Type") or "").lower()
            if "json" not in ctype:
                snippet = (r.text or "")[:200].replace("\n", " ")
                raise ValueError(f"Non-JSON response from {url} (Content-Type={ctype}): {snippet}")

            return r.json()

        except Exception as e:
            last_err = e
            sleep_s = min(60, (2 ** attempt)) + random.uniform(0.0, 0.75)
            print(f"Overpass attempt {attempt+1}/{max_tries} failed ({e}). Sleep {sleep_s:.1f}s; next mirror...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Overpass failed after {max_tries} tries. Last error: {last_err}")


def count_tourism_business_cached(sid, lat, lon, radius_m):
    """Counts tourism/business proxies around a point. Cached by station+radius."""
    cache_path = _cache_path_for_station(sid, radius_m)

    # cache hit
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            row = json.load(f)
        # Old cache files may not contain radius_m -> inject it
        if "radius_m" not in row:
            row["radius_m"] = radius_m
        return row

    query = f"""
    [out:json][timeout:180];
    (
      node["tourism"="hotel"](around:{radius_m},{lat},{lon});
      node["tourism"="hostel"](around:{radius_m},{lat},{lon});
      node["tourism"="guest_house"](around:{radius_m},{lat},{lon});

      node["office"](around:{radius_m},{lat},{lon});
      way["office"](around:{radius_m},{lat},{lon});

      way["landuse"="industrial"](around:{radius_m},{lat},{lon});
      way["man_made"="works"](around:{radius_m},{lat},{lon});
      way["building"="warehouse"](around:{radius_m},{lat},{lon});
    );
    out tags;
    """

    data = overpass_post_with_retries(query, timeout=120, max_tries=6)
    elements = data.get("elements", [])

    counts = {
        "idEstacion": sid,
        "radius_m": radius_m,
        "hotels": 0,
        "hostels": 0,
        "guest_houses": 0,
        "offices": 0,
        "industrial": 0,
        "factories": 0,
        "warehouses": 0,
    }

    for e in elements:
        tags = e.get("tags", {})

        # tourism
        t = tags.get("tourism")
        if t == "hotel":
            counts["hotels"] += 1
        elif t == "hostel":
            counts["hostels"] += 1
        elif t == "guest_house":
            counts["guest_houses"] += 1

        # business / industrial proxies
        if "office" in tags:
            counts["offices"] += 1
        if tags.get("landuse") == "industrial":
            counts["industrial"] += 1
        if tags.get("man_made") == "works":
            counts["factories"] += 1
        if tags.get("building") == "warehouse":
            counts["warehouses"] += 1

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False)

    time.sleep(BASE_SLEEP)
    return counts


def rebuild_overpass_from_cache(cache_dir):
    """
    Rebuild a DataFrame from cache files.
    Handles old cache JSONs that don't have radius_m by parsing from filename:
      osm_<sid>_<radius>m.json
    """
    pat = re.compile(r"^osm_(\d+)_(\d+)m\.json$")
    rows = []

    for fn in os.listdir(cache_dir):
        m = pat.match(fn)
        if not m:
            continue
        sid_from_name = int(m.group(1))
        radius_from_name = int(m.group(2))

        with open(os.path.join(cache_dir, fn), "r", encoding="utf-8") as f:
            row = json.load(f)

        row["idEstacion"] = row.get("idEstacion", sid_from_name)
        row["radius_m"] = row.get("radius_m", radius_from_name)
        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        for c in ["hotels", "hostels", "guest_houses", "offices", "industrial", "factories", "warehouses"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        df["tourism_index"] = df["hotels"] + df["hostels"] + df["guest_houses"]
        df["business_index"] = df["offices"] + df["industrial"] + df["factories"] + df["warehouses"]

        df = df.drop_duplicates(subset=["idEstacion", "radius_m"]).sort_values(["idEstacion", "radius_m"])

    return df


# =========================
# 1) PULL STATIONS (PRECIOIL)
# =========================
all_stations = []
seen_ids = set()

for name, lat, lon in anchors:
    batch = get_stations_in_radius(lat, lon, radio_km=25, limite=50, max_paginas=30)
    print(name, "raw stations returned:", len(batch))
    for s in batch:
        sid = s.get("idEstacion")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            all_stations.append(s)

stations = all_stations
all_station_ids = sorted(seen_ids)

print("Total unique stations:", len(all_station_ids))


# =========================
# 2) STATION META
# =========================
station_meta = pd.DataFrame(stations)

keep_cols = [
    "idEstacion", "marca", "nombreEstacion",
    "latitud", "longitud",
    "nombreMunicipio", "provincia", "codPostal",
    "distancia"
]
station_meta = station_meta[[c for c in keep_cols if c in station_meta.columns]].copy()

for col in ["latitud", "longitud"]:
    if col in station_meta.columns:
        station_meta[col] = station_meta[col].astype(str).str.replace(",", ".", regex=False)
        station_meta[col] = pd.to_numeric(station_meta[col], errors="coerce")

station_meta = station_meta.drop_duplicates(subset=["idEstacion"])
print("Station meta shape:", station_meta.shape)
print(station_meta[["idEstacion", "latitud", "longitud"]].head())
if "distancia" in station_meta.columns:
    print(station_meta["distancia"].describe())


# =========================
# 3) DAILY PRICE PANEL (G95)
# =========================
frames = []
failed = []

for i, sid in enumerate(all_station_ids, start=1):
    try:
        hist = fetch_history_cached(sid, FECHA_INICIO, FECHA_FIN)
        if isinstance(hist, dict) and "errors" in hist:
            failed.append((sid, hist["errors"]))
            continue

        daily = history_to_daily_g95(hist)
        if not daily.empty:
            frames.append(daily)

    except Exception as e:
        failed.append((sid, str(e)))

    if i % 25 == 0:
        print(f"Processed {i}/{len(all_station_ids)} stations...")

panel_daily_g95 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print("Final panel shape (before merge):", panel_daily_g95.shape)

if panel_daily_g95.empty:
    print("No daily data returned for this window. Check date range or fuel availability.")
    print("Example failed stations:", failed[:3])
    raise SystemExit

panel_daily_g95 = panel_daily_g95.merge(station_meta, on="idEstacion", how="left")

print("Merged panel shape:", panel_daily_g95.shape)
print("Earliest date in panel:", panel_daily_g95["date"].min())
print("Latest date in panel:", panel_daily_g95["date"].max())
print("Total unique dates:", panel_daily_g95["date"].nunique())

print("Failed stations:", len(failed))
if failed[:2]:
    print("Example failures:", failed[:2])

# Save the price panel (OVERWRITE)
write_csv_overwrite(panel_daily_g95, OUT_PANEL_CSV)


# =========================
# 3B) COMPETITION PROXY (2km)
# =========================
# Produces station-week panel with:
#  - local_price_variance_2km / std using neighbor weekly mean prices
#  - avg_neighbor_change_freq_2km (teacher metric): compute EACH neighbor's weekly change freq then average
# Also writes station-level averages across weeks.
print("\n=== Building competition proxy: local price variance + teacher neighbor change frequency (2km) ===")

# Clean types for weekly computations
df = panel_daily_g95.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["precio_last"] = pd.to_numeric(df["precio_last"], errors="coerce")
df = df.dropna(subset=["idEstacion", "date", "precio_last", "latitud", "longitud"]).copy()

# Weekly identifiers
iso = df["date"].dt.isocalendar()
df["year"] = iso.year.astype(int)
df["week"] = iso.week.astype(int)

# Weekly mean price for ALL stations
weekly_prices = (
    df.groupby(["idEstacion", "year", "week"], as_index=False)
      .agg(
          weekly_mean_price=("precio_last", "mean"),
          days_obs=("precio_last", "count")
      )
)

# Choose sample of stations
meta_coords = station_meta.dropna(subset=["latitud", "longitud"]).copy()
if "distancia" in meta_coords.columns:
    meta_coords["distancia"] = pd.to_numeric(meta_coords["distancia"], errors="coerce")
    sample_meta = meta_coords.sort_values("distancia").head(SAMPLE_N_STATIONS).copy()
else:
    sample_meta = meta_coords.head(SAMPLE_N_STATIONS).copy()

sample_ids = sample_meta["idEstacion"].tolist()
print("Sample stations for competition:", len(sample_ids))

# Neighbor mapping (neighbors from ALL stations with coords)
radius_km = COMP_RADIUS_M / 1000.0
all_coords = meta_coords[["idEstacion", "latitud", "longitud"]].copy()

neighbors = {}
for _, r in sample_meta.iterrows():
    sid_i = r["idEstacion"]
    lat_i = float(r["latitud"])
    lon_i = float(r["longitud"])

    neigh = []
    for _, s in all_coords.iterrows():
        sid_j = s["idEstacion"]
        if sid_j == sid_i:
            continue
        lat_j = float(s["latitud"])
        lon_j = float(s["longitud"])

        if haversine_km(lat_i, lon_i, lat_j, lon_j) <= radius_km:
            neigh.append(sid_j)

    neighbors[sid_i] = neigh

# Diagnostics
n_counts = pd.Series({k: len(v) for k, v in neighbors.items()})
print("Neighbor counts (2km) summary for sampled stations:")
print(n_counts.describe())

# --- Teacher metric prep: weekly change frequency per station ---
df = df.sort_values(["idEstacion", "date"]).copy()
df["price_change"] = df.groupby("idEstacion")["precio_last"].diff()
df["changed"] = (df["price_change"].fillna(0) != 0).astype(int)

weekly_changefreq = (
    df.groupby(["idEstacion", "year", "week"], as_index=False)
      .agg(
          change_freq=("changed", "mean"),
          days_in_week=("changed", "size")
      )
)

freq_key = weekly_changefreq.set_index(["idEstacion", "year", "week"])
weekly_key = weekly_prices.set_index(["idEstacion", "year", "week"])

rows = []
for sid_i in sample_ids:
    neigh_ids = neighbors.get(sid_i, [])
    if not neigh_ids:
        continue

    weeks_i = weekly_prices[weekly_prices["idEstacion"] == sid_i][["year", "week"]].values.tolist()

    for year, week in weeks_i:
        # Teacher metric: avg over neighbors of (neighbor's change_freq in that week)
        neigh_freqs = []
        for sid_j in neigh_ids:
            kk = (sid_j, int(year), int(week))
            if kk in freq_key.index:
                neigh_freqs.append(float(freq_key.loc[kk, "change_freq"]))
        avg_neighbor_change_freq = float("nan") if len(neigh_freqs) == 0 else float(pd.Series(neigh_freqs).mean())

        # Neighbor weekly mean prices -> variance/std
        neigh_prices = []
        for sid_j in neigh_ids:
            k = (sid_j, int(year), int(week))
            if k in weekly_key.index:
                neigh_prices.append(float(weekly_key.loc[k, "weekly_mean_price"]))

        if len(neigh_prices) >= MIN_NEIGHBORS_WITH_PRICE:
            local_var = float(pd.Series(neigh_prices).var())
            local_std = float(pd.Series(neigh_prices).std())
        else:
            local_var = float("nan")
            local_std = float("nan")

        rows.append({
            "idEstacion": sid_i,
            "year": int(year),
            "week": int(week),
            "local_price_variance_2km": local_var,
            "local_price_std_2km": local_std,
            "num_neighbors_total_2km": len(neigh_ids),
            "num_neighbors_with_price_that_week": len(neigh_prices),
            "avg_neighbor_change_freq_2km": avg_neighbor_change_freq,
            "num_neighbors_with_freq_that_week": len(neigh_freqs),
        })

competition_weekly = pd.DataFrame(rows)
print("Competition weekly panel shape:", competition_weekly.shape)
print(competition_weekly.head())

# OVERWRITE station-week competition file
write_csv_overwrite(competition_weekly, OUT_COMP_CSV)

# ALSO write station-level AVERAGES across weeks (easy Excel check)
comp_station_avg = (
    competition_weekly.groupby("idEstacion", as_index=False)
      .agg(
          avg_local_price_variance_2km=("local_price_variance_2km", "mean"),
          avg_local_price_std_2km=("local_price_std_2km", "mean"),
          avg_neighbor_change_freq_2km=("avg_neighbor_change_freq_2km", "mean"),
          avg_num_neighbors_with_price=("num_neighbors_with_price_that_week", "mean"),
          avg_num_neighbors_with_freq=("num_neighbors_with_freq_that_week", "mean"),
          n_station_weeks=("week", "count"),
      )
)
write_csv_overwrite(comp_station_avg, OUT_COMP_AVG_CSV)


# =========================
# 4) OVERPASS FEATURES (FETCH REMAINING FOR THIS RADIUS)
# =========================
eligible = station_meta.dropna(subset=["latitud", "longitud"]).copy()

def has_cache(sid):
    return os.path.exists(_cache_path_for_station(sid, RADIUS_M))

eligible["has_cache"] = eligible["idEstacion"].apply(has_cache)
already_done = int(eligible["has_cache"].sum())
remaining_df = eligible[~eligible["has_cache"]].copy()

print("\n=== Overpass run ===")
print("Stations total (with coords):", len(eligible))
print("Already cached for radius", RADIUS_M, "m:", already_done)
print("Remaining to fetch:", len(remaining_df))

if RUN_SAMPLE:
    n = min(TARGET_N, len(remaining_df))
    to_process = remaining_df.sample(n, random_state=42).reset_index(drop=True) if n > 0 else remaining_df.head(0)
else:
    to_process = remaining_df.reset_index(drop=True)

print("This run will process:", len(to_process), "stations (OSM)")

area_rows = []
fail_osm = []

for idx, row in to_process.iterrows():
    sid = row["idEstacion"]
    lat = row["latitud"]
    lon = row["longitud"]

    if has_cache(sid):
        continue

    try:
        counts = count_tourism_business_cached(sid, lat, lon, radius_m=RADIUS_M)
        area_rows.append(counts)
    except Exception as e:
        fail_osm.append((sid, str(e)))
        print("Final failure for station", sid, ":", e)

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(to_process)} stations (OSM)")

new_area_df = pd.DataFrame(area_rows)
print("New rows returned this run:", len(new_area_df))
if not new_area_df.empty:
    new_area_df["tourism_index"] = new_area_df["hotels"] + new_area_df["hostels"] + new_area_df["guest_houses"]
    new_area_df["business_index"] = new_area_df["offices"] + new_area_df["industrial"] + new_area_df["factories"] + new_area_df["warehouses"]

# =========================
# 5) BUILD FINAL OSM FEATURES TABLE
# =========================
cached_df = rebuild_overpass_from_cache(OVERPASS_CACHE_DIR)

if not new_area_df.empty:
    combined = pd.concat([cached_df, new_area_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["idEstacion", "radius_m"]).sort_values(["idEstacion", "radius_m"])
else:
    combined = cached_df

print("Total rows with OSM features (all radii cached):", len(combined))

# OVERWRITE OSM features file
write_csv_overwrite(combined, OUT_OSM_CSV)

print("Failures this run:", len(fail_osm))
if fail_osm[:3]:
    print("Example failures:", fail_osm[:3])

print("OSM CSV exists?", os.path.exists(OUT_OSM_CSV), "| size bytes:", (os.path.getsize(OUT_OSM_CSV) if os.path.exists(OUT_OSM_CSV) else 0))
print("Cache folder exists?", os.path.exists(OVERPASS_CACHE_DIR), "| cache files:", len(os.listdir(OVERPASS_CACHE_DIR)))

print("\nDone.")
print("Outputs (OVERWRITTEN each run unless locked):")
print(" -", OUT_PANEL_CSV)
print(" -", OUT_COMP_CSV)
print(" -", OUT_COMP_AVG_CSV)
print(" -", OUT_OSM_CSV)
