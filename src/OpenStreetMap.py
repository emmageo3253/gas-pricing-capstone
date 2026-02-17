# =========================
# FULL WORKING SCRIPT (IDLE)
# Precioil -> build station_meta + (optional) price panel
# Then: build neighbor station_ids using Precioil coordinates (no Overpass needed)
# =========================

import os, json, time, math
import requests
import pandas as pd

# -------------------------
# PRECIOIL SETTINGS
# -------------------------
BASE_URL = "https://api.precioil.es"

# Date window for price history (optional part below)
FECHA_INICIO = "2026-01-01"
FECHA_FIN    = "2026-01-31"

G95_CODE = 6
SLEEP_SECONDS = 0.35

CACHE_DIR = "cache_hist"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# STATION PULL SETTINGS (Madrid example)
# Add anchors to expand number of stations
# -------------------------
anchors = [
    ("Madrid_Center", 40.4168, -3.7038),
    ("Madrid_North",  40.5000, -3.7000),
    ("Madrid_South",  40.3500, -3.7000),
    ("Madrid_East",   40.4168, -3.6000),
    ("Madrid_West",   40.4168, -3.8200),
]

RADIO_KM = 25
LIMITE = 50
MAX_PAGINAS = 30

# -------------------------
# NEIGHBOR SETTINGS (competition set)
# -------------------------
NEIGHBOR_RADIUS_KM = 2.0   # try 2km first; you can also do 3, 5, etc.
MAX_NEIGHBORS = 20         # cap number of neighbors per station


# ============================================================
# 1) PRECIOIL: get stations in radius (pagination + dedupe)
# ============================================================
def get_stations_in_radius(lat, lon, radio_km=15, limite=50, max_paginas=20):
    seen = set()
    stations = []

    for pagina in range(1, max_paginas + 1):
        params = {
            "latitud": lat,
            "longitud": lon,
            "radio": radio_km,
            "pagina": pagina,
            "limite": limite
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


# ============================================================
# 2) Build station_meta (THIS fixes your station_meta error)
# ============================================================
print("Pulling stations from Precioil...")
all_stations = []
seen_ids = set()

for name, lat, lon in anchors:
    batch = get_stations_in_radius(lat, lon, radio_km=RADIO_KM, limite=LIMITE, max_paginas=MAX_PAGINAS)
    print(name, "raw stations returned:", len(batch))
    for s in batch:
        sid = s.get("idEstacion")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            all_stations.append(s)

stations = all_stations
print("Total unique stations:", len(stations))

station_meta = pd.DataFrame(stations)

# Keep useful columns (if present)
keep_cols = [
    "idEstacion", "marca", "nombreEstacion",
    "latitud", "longitud",
    "nombreMunicipio", "provincia", "codPostal",
    "distancia"
]
station_meta = station_meta[[c for c in keep_cols if c in station_meta.columns]].copy()

# Normalize lat/long to floats
for col in ["latitud", "longitud"]:
    station_meta[col] = station_meta[col].astype(str).str.replace(",", ".", regex=False)
    station_meta[col] = pd.to_numeric(station_meta[col], errors="coerce")

station_meta["idEstacion"] = pd.to_numeric(station_meta["idEstacion"], errors="coerce")
station_meta = station_meta.dropna(subset=["idEstacion", "latitud", "longitud"]).drop_duplicates(subset=["idEstacion"])
station_meta["idEstacion"] = station_meta["idEstacion"].astype(int)

print("station_meta shape:", station_meta.shape)
print(station_meta[["idEstacion","latitud","longitud"]].head())

station_meta.to_csv("station_meta.csv", index=False)
print("Saved: station_meta.csv")


# ============================================================
# 3) Build neighbor station IDs directly from Precioil coordinates
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

print("\nBuilding neighbor graph from Precioil coordinates...")
df = station_meta.copy()

neighbor_edges = []

# Convert to lists for speed (IDLE-friendly)
ids = df["idEstacion"].tolist()
lats = df["latitud"].tolist()
lons = df["longitud"].tolist()

for i in range(len(ids)):
    sid = ids[i]
    lat1 = lats[i]
    lon1 = lons[i]

    dists = []
    for j in range(len(ids)):
        if i == j:
            continue
        d = haversine_km(lat1, lon1, lats[j], lons[j])
        if d <= NEIGHBOR_RADIUS_KM:
            dists.append((ids[j], d))

    dists.sort(key=lambda x: x[1])
    dists = dists[:MAX_NEIGHBORS]

    for nb_id, d in dists:
        neighbor_edges.append({
            "idEstacion": sid,
            "neighbor_idEstacion": nb_id,
            "distance_km": d,
            "radius_km": NEIGHBOR_RADIUS_KM
        })

neighbor_df = pd.DataFrame(neighbor_edges)
print("Neighbor edges:", len(neighbor_df))
print(neighbor_df.head())

neighbor_df.to_csv("station_neighbors_precioil_only.csv", index=False)
print("Saved: station_neighbors_precioil_only.csv")
