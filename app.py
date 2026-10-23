"""
Dubai Property Price Estimator - local web demo
PDA2 project, Ajman University, Fall 2026

Runs the trained Keras model (Final A) on your own laptop:
    python app.py      ->  open http://127.0.0.1:5000

The page sends the property details to /api/predict, which applies the SAME
steps as the notebook:
    1. date_listed -> listing_year, listing_month
    2. villas -> floor = 0 and total_floors = 0
    3. the saved preprocessor (one-hot + StandardScaler fitted on the training set)
    4. the network predicts log(price), and exp() converts it back to USD
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up messages

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from tensorflow import keras

import photos                                       # community photos for the Find homes cards (photos.py)
import metro                                        # Dubai Metro lines for the map's Metro switch (metro.py)
import gzip
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load the trained model and the preprocessor once, when the app starts
# ---------------------------------------------------------------------------
model = keras.models.load_model(os.path.join(HERE, "dubai_price_model.keras"))
preprocessor = joblib.load(os.path.join(HERE, "preprocessor.joblib"))
FEATURES = list(preprocessor.feature_names_in_)     # the 25 training columns, in order

# ---------------------------------------------------------------------------
# 2. Lookup tables built from the dataset
#    The user only picks a community; its location details (zone, freehold
#    status, coordinates, distance to Burj Khalifa, usual metro station) are
#    filled in from the typical values for that community in the data.
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "secondary_sales.csv"))
df["month"] = df["date_listed"].str[:7]               # "2024-11"


def most_common(s):
    return s.value_counts().index[0]


COMMUNITIES = {}
for name, g in df.groupby("community"):
    station = most_common(g["metro_station"])
    s = g[g["metro_station"] == station]
    access = most_common(s["metro_distance_type"])
    COMMUNITIES[name] = {
        "zone": g["zone"].iloc[0],
        "is_freehold": bool(g["is_freehold"].iloc[0]),
        "lat": float(g["lat"].median()),
        "lon": float(g["lon"].median()),
        "to_burj_khalifa_km": round(float(g["to_burj_khalifa_km"].median()), 2),
        "categories": sorted(g["property_category"].unique().tolist()),
        "metro_station": station,
        "metro_distance_type": access,
        "metro_distance_min": int(s.loc[s["metro_distance_type"] == access, "metro_distance_min"].median()),
    }

METRO_LINE = df.groupby("metro_station")["metro_line"].first().to_dict()

# Price per sq ft is used only for the charts and the map, never as a model input
# (it is calculated from the price, so using it in the model would be data leakage).
df["quarter"] = pd.to_datetime(df["date_listed"]).dt.to_period("Q").astype(str)    # "2024Q3"
DUBAI_PPSF = float(df["price_per_sqft_usd"].median())
for name, g in df.groupby("community"):
    COMMUNITIES[name]["ppsf"] = int(g["price_per_sqft_usd"].median())
    COMMUNITIES[name]["sales"] = int(len(g))
QUARTERS = sorted(df["quarter"].unique())
DUBAI_TREND = df.groupby("quarter")["price_per_sqft_usd"].median()
COMMUNITY_TREND = df.groupby(["community", "quarter"])["price_per_sqft_usd"].agg(["median", "size"])

# The mortgage rate is a market figure for the listing month, not a property feature
MORTGAGE_RATE = df.groupby("month")["mortgage_rate_at_listing"].median().to_dict()
MONTHS = sorted(MORTGAGE_RATE)                         # "2020-01" ... "2026-04"
LATEST_MONTH = MONTHS[-1]                              # estimates use the latest market data

# Typical building height per community. The website does not ask for it:
# replacing the real value with this one changed the error by 0.01 points (9.54% -> 9.53%).
TYPICAL_FLOORS = (df[df["property_category"] == "apartment"]
                  .groupby("community")["total_floors"].median().to_dict())

TYPES = {
    "apartment": ["studio", "1BR", "2BR", "3BR", "4BR_penthouse"],
    "villa": ["3BR_villa", "4BR_villa", "5BR_villa", "6BR_villa"],
}
BEDROOMS = {"studio": 0, "1BR": 1, "2BR": 2, "3BR": 3, "4BR_penthouse": 4,
            "3BR_villa": 3, "4BR_villa": 4, "5BR_villa": 5, "6BR_villa": 6}

OPTIONS = {
    "communities": COMMUNITIES,
    "types": TYPES,
    "views": sorted(df["view"].unique().tolist()),
    "furnishing": sorted(df["furnishing"].unique().tolist()),
    "conditions": sorted(df["condition"].unique().tolist()),
    "stations": sorted(METRO_LINE),
    "months": MONTHS,
    "latest_month": LATEST_MONTH,
    "dubai_ppsf": round(DUBAI_PPSF),
    "rates": MORTGAGE_RATE,
    # typical size of each property type in each community, used to pre-fill the area box
    "typical_area": {c: g.groupby("property_type")["area_sqft"].median().round().astype(int).to_dict()
                     for c, g in df.groupby("community")},
}


# ---------------------------------------------------------------------------
# 3. Prediction: the same pipeline as the notebook
# ---------------------------------------------------------------------------
def build_row(form: dict) -> pd.DataFrame:
    """Turn the form answers into one row with the 25 training columns."""
    c = COMMUNITIES[form["community"]]
    ptype = form["property_type"]
    category = "villa" if ptype.endswith("_villa") else "apartment"
    listing_month = form.get("listing_month") or LATEST_MONTH
    year, month = (int(x) for x in listing_month.split("-"))
    floor = float(form.get("floor") or 0)
    total_floors = form.get("total_floors")
    total_floors = float(total_floors) if total_floors else max(TYPICAL_FLOORS.get(form["community"], 40.0), floor)
    station = form.get("metro_station") or c["metro_station"]

    row = {
        "community": form["community"],
        "zone": c["zone"],
        "is_freehold": c["is_freehold"],
        "lat": c["lat"],
        "lon": c["lon"],
        "property_category": category,
        "property_type": ptype,
        "bedrooms": BEDROOMS[ptype],
        "area_sqft": float(form["area_sqft"]),
        "floor": floor,
        "total_floors": total_floors,
        "year_built": int(form["year_built"]),
        "view": form["view"],
        "furnishing": form["furnishing"],
        "condition": form["condition"],
        "parking_spaces": int(form["parking_spaces"]),
        "chiller_included": bool(form.get("chiller_included")),
        "metro_station": station,
        "metro_line": METRO_LINE[station],
        "metro_distance_min": int(form.get("metro_distance_min") or c["metro_distance_min"]),
        "metro_distance_type": form.get("metro_distance_type") or c["metro_distance_type"],
        "to_burj_khalifa_km": c["to_burj_khalifa_km"],
        "mortgage_rate_at_listing": MORTGAGE_RATE[listing_month],
        "listing_year": year,
        "listing_month": month,
    }

    # Same rule as the notebook: floor numbers do not apply to villas
    if category == "villa":
        row["floor"] = 0.0
        row["total_floors"] = 0.0

    return pd.DataFrame([row])[FEATURES]


def predict_price(X: pd.DataFrame) -> np.ndarray:
    X_p = preprocessor.transform(X).astype("float32")      # 235 inputs
    # Calling the network directly is ~25x faster than model.predict() for a few rows
    # (predict() sets up a whole data pipeline each time, which costs ~60 ms per call).
    log_price = model(X_p, training=False).numpy().ravel()  # the network predicts log(price)
    return np.exp(log_price)                               # back to USD


@lru_cache(maxsize=2048)                             # the sales data never changes while the app runs
def similar_sales(community: str, ptype: str):
    """Real sales of the same type in the same community, for comparison."""
    s = df[(df["community"] == community) & (df["property_type"] == ptype)]["price_usd"]
    if len(s) < 5:
        return None
    return {"count": int(len(s)), "median": int(s.median()),
            "low": int(s.quantile(0.25)), "high": int(s.quantile(0.75))}


@lru_cache(maxsize=2048)                             # the sales data never changes while the app runs
def comparable_sales(community: str, ptype: str, area: float, n: int = 5):
    """The n real sales most similar to the property: same community and type, closest size."""
    s = df[(df["community"] == community) & (df["property_type"] == ptype)].copy()
    s["gap"] = (s["area_sqft"] - area).abs()
    s = s.nsmallest(n, "gap").sort_values("date_listed", ascending=False)
    return [{"date": r.date_listed[:7], "area_sqft": int(r.area_sqft),
             "floor": None if pd.isna(r.floor) else int(r.floor),
             "view": r.view, "furnishing": r.furnishing, "price_usd": int(r.price_usd)}
            for r in s.itertuples()]


@lru_cache(maxsize=2048)                             # the sales data never changes while the app runs
def price_trend(community: str):
    """Quarterly median price per sq ft for the community and for all of Dubai."""
    ct = COMMUNITY_TREND.loc[community]
    comm = [float(ct.loc[q, "median"]) if q in ct.index and ct.loc[q, "size"] >= 5 else None for q in QUARTERS]
    years = df[df["community"] == community].groupby(df["date_listed"].str[:4])["price_per_sqft_usd"].median()
    return {
        "quarters": QUARTERS,
        "community": comm,
        "dubai": [float(DUBAI_TREND[q]) for q in QUARTERS],
        "first_year": years.index[0], "last_year": years.index[-1],
        "change_pct": round((years.iloc[-1] / years.iloc[0] - 1) * 100),
        "vs_dubai_pct": round((COMMUNITIES[community]["ppsf"] / DUBAI_PPSF - 1) * 100),
    }


def what_if(form: dict, base_price: float):
    """Re-run the model with one detail changed at a time, to show what moves the price."""
    variants = []
    for field, values in [("view", OPTIONS["views"]), ("furnishing", OPTIONS["furnishing"]),
                          ("condition", OPTIONS["conditions"])]:
        variants += [(field, v) for v in values if v != form[field]]
    p = int(form["parking_spaces"])
    variants += [("parking_spaces", str(x)) for x in (p - 1, p + 1) if 1 <= x <= 4]
    y = int(form["year_built"])
    variants += [("year_built", str(x)) for x in (y - 5, y + 5) if 2008 <= x <= 2023]
    variants.append(("chiller_included", not bool(form.get("chiller_included"))))

    rows = [build_row({**form, field: value}) for field, value in variants]
    prices = predict_price(pd.concat(rows))                 # one batch: fast

    best = {}                                               # per field: biggest gain and biggest loss
    for (field, value), price in zip(variants, prices):
        delta = float(price - base_price)
        if abs(delta) < 0.005 * base_price:                 # ignore changes under 0.5%
            continue
        key = (field, delta > 0)
        if key not in best or abs(delta) > abs(best[key]["delta_usd"]):
            best[key] = {"field": field, "value": value, "delta_usd": round(delta)}
    ups = sorted([v for (f, up), v in best.items() if up], key=lambda v: -v["delta_usd"])[:3]
    downs = sorted([v for (f, up), v in best.items() if not up], key=lambda v: v["delta_usd"])[:2]
    return ups + downs


# ---------------------------------------------------------------------------
# 3b. Market figures used by the "Find homes" and "Sell" pages
# ---------------------------------------------------------------------------
RECENT_FROM = "2025-01"                                    # "recent sales" = 2025 and 2026
RECENT = df[df["date_listed"] >= RECENT_FROM]

# One-year price change per community: median price per sq ft over the last 4 quarters
# compared with the 4 quarters before them.
_last4, _prev4 = QUARTERS[-4:], QUARTERS[-8:-4]
_now = df[df["quarter"].isin(_last4)].groupby("community")["price_per_sqft_usd"].median()
_before = df[df["quarter"].isin(_prev4)].groupby("community")["price_per_sqft_usd"].median()
COMMUNITY_YOY = ((_now / _before - 1) * 100).round(1).dropna().to_dict()

# Extra figures for the map's hover card: 1-year change, the last 3 years of quarterly
# price per sq ft (for the small line chart) and the typical price of a common home there.
for name, g in df.groupby("community"):
    c = COMMUNITIES[name]
    c["yoy"] = COMMUNITY_YOY.get(name)
    ct = COMMUNITY_TREND.loc[name]
    c["spark"] = [int(ct.loc[q, "median"]) if q in ct.index and ct.loc[q, "size"] >= 5 else None for q in QUARTERS[-12:]]
    recent = g[g["date_listed"] >= RECENT_FROM]
    recent = recent if len(recent) >= 20 else g
    counts = recent["property_type"].value_counts()
    ptype = "2BR" if counts.get("2BR", 0) >= 5 else counts.index[0]          # 2-bed if there are enough, else the usual type
    c["typical"] = {"type": ptype, "price": int(recent.loc[recent["property_type"] == ptype, "price_usd"].median())}
    c["median_price"] = int(recent["price_usd"].median())                    # all homes sold there recently
OPTIONS["spark_quarters"] = [QUARTERS[-12], QUARTERS[-1]]                  # first and last quarter of the small charts

# Seasonal pattern for all of Dubai: each month's median price per sq ft divided by the
# 13-month centred median (this removes the long-term trend), averaged by season.
_monthly = df.groupby("month")["price_per_sqft_usd"].median()
_ratio = (_monthly / _monthly.rolling(13, center=True).median()).dropna()
_by_month = _ratio.groupby(_ratio.index.str[5:].astype(int)).mean()
SEASON_MONTHS = {"winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11]}
SEASONS = {k: round(float((_by_month[m].mean() - 1) * 100), 1) for k, m in SEASON_MONTHS.items()}


def search_homes(q: dict):
    """Recent sales (2025-2026) that fit a budget, optionally in one community / type / bedrooms."""
    s = RECENT
    if q.get("category") in ("apartment", "villa"):
        s = s[s["property_category"] == q["category"]]
    if int(q.get("bedrooms_min") or 0) > 0:
        s = s[s["bedrooms"] >= int(q["bedrooms_min"])]
    community = q.get("community")
    # price histogram of the homes matching the other filters (drawn behind the price slider)
    in_area = s[s["community"] == community] if community else s
    hist = np.histogram(in_area["price_usd"].clip(PRICE_EDGES[0], PRICE_EDGES[-1]), bins=PRICE_EDGES)[0]
    if float(q.get("budget_max_usd") or 0) > 0:
        s = s[s["price_usd"] <= float(q["budget_max_usd"])]
    if float(q.get("budget_min_usd") or 0) > 0:
        s = s[s["price_usd"] >= float(q["budget_min_usd"])]
    # matching homes in every community (for the map), before narrowing to one community
    per_area = s.groupby("community")["price_usd"].agg(["size", "median"])
    if community:
        s = s[s["community"] == community]

    by = (s.groupby("community")
            .agg(count=("price_usd", "size"), median_price=("price_usd", "median"), median_area=("area_sqft", "median"))
            .sort_values("count", ascending=False).head(6).reset_index())
    order = {"newest": ("date_listed", False), "price_asc": ("price_usd", True),
             "price_desc": ("price_usd", False), "size_desc": ("area_sqft", False)}
    col, asc = order.get(q.get("sort"), order["newest"])
    page, size = int(q.get("page") or 0), 12
    rows = s.sort_values(col, ascending=asc).iloc[page * size:(page + 1) * size]
    return {
        "total": int(len(s)),
        "hist": [int(v) for v in hist],
        "map": {c: [int(r["size"]), int(r["median"])] for c, r in per_area.iterrows()},   # community -> [homes, median price]
        "communities": [{"community": r.community, "count": int(r["count"]), "median_price": int(r.median_price),
                         "median_area": int(r.median_area)} for _, r in by.iterrows()],
        "homes": [{"community": r.community, "property_type": r.property_type, "area_sqft": int(r.area_sqft),
                   "floor": None if pd.isna(r.floor) else int(r.floor), "year_built": int(r.year_built),
                   "view": r.view, "furnishing": r.furnishing, "condition": r.condition,
                   "parking_spaces": int(r.parking_spaces), "chiller_included": bool(r.chiller_included),
                   "price_usd": int(r.price_usd), "date": r.date_listed[:7]} for r in rows.itertuples()],
    }


def sell_advice(form: dict):
    """Market value, asking-price check, profit and timing advice for a property the user owns."""
    X = build_row(form)
    value = float(predict_price(X)[0])
    community = form["community"]
    comps = comparable_sales(community, form["property_type"], float(form["area_sqft"]), n=6)

    asking = float(form.get("asking_price_usd") or 0)
    bought = float(form.get("purchase_price_usd") or 0)
    bought_year = int(form.get("purchase_year") or 0)
    years_held = max(0.5, int(LATEST_MONTH[:4]) - bought_year + 0.5) if bought_year else None

    yoy = COMMUNITY_YOY.get(community, 0.0)
    best_season = max(SEASONS, key=SEASONS.get)
    spread = max(SEASONS.values()) - min(SEASONS.values())
    if yoy >= 5:
        timing = "hold"            # prices still rising clearly: waiting may add value
    elif yoy <= -3:
        timing = "now"             # prices softening: selling sooner is safer
    else:
        timing = "season"          # stable market: sell now or in the strongest season

    return {
        "value_usd": round(value), "range_10": [round(value * 0.9), round(value * 1.1)],
        "comparables": comps,
        "asking_usd": round(asking) or None,
        "asking_diff_pct": round((asking / value - 1) * 100, 1) if asking else None,
        "profit_usd": round(value - bought) if bought else None,
        "profit_pct": round((value / bought - 1) * 100, 1) if bought else None,
        "annual_pct": round(((value / bought) ** (1 / years_held) - 1) * 100, 1) if bought and years_held else None,
        "yoy_pct": yoy, "timing": timing, "best_season": best_season,
        "seasons": SEASONS, "season_spread": round(spread, 1),
    }


# ---------------------------------------------------------------------------
# 4. Exchange rates for the currency selector
#    The model predicts in USD. Live rates are fetched at most every 6 hours
#    from a free API (no key needed); if the laptop is offline, the fixed
#    rates below (14 Oct 2026) are used instead.
# ---------------------------------------------------------------------------
CURRENCIES = ["USD", "AED", "SAR", "QAR", "KWD", "BHD", "OMR", "EUR", "GBP",
              "INR", "PKR", "EGP", "TRY", "CNY", "JPY", "CAD", "AUD", "CHF", "RUB"]

FALLBACK_RATES = {
    "USD": 1, "AED": 3.6725, "SAR": 3.75, "QAR": 3.64, "KWD": 0.31129, "BHD": 0.376,
    "OMR": 0.384497, "EUR": 0.893421, "GBP": 0.754721, "INR": 95.319179,
    "PKR": 277.05126, "EGP": 52.804385, "TRY": 49.379871, "CNY": 6.705025,
    "JPY": 158.307574, "CAD": 1.422113, "AUD": 1.432809, "CHF": 0.832849, "RUB": 83.733665,
}
_rates_cache = {"rates": FALLBACK_RATES, "updated": "2026-10-14", "live": False, "fetched": 0.0, "tried": 0.0}


def get_rates():
    import json
    import time
    import urllib.request

    now = time.time()
    if _rates_cache["live"] and now - _rates_cache["fetched"] < 6 * 3600:
        return _rates_cache
    if now - _rates_cache["tried"] < 600:          # offline: retry at most every 10 minutes
        return _rates_cache
    _rates_cache["tried"] = now
    try:
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=4) as r:
            data = json.load(r)
        rates = {c: data["rates"][c] for c in CURRENCIES if c in data["rates"]}
        updated = time.strftime("%Y-%m-%d", time.gmtime(data["time_last_update_unix"]))
        _rates_cache.update(rates=rates, updated=updated, live=True, fetched=time.time())
    except Exception:
        pass                                   # offline: keep the fallback rates
    return _rates_cache


# ---------------------------------------------------------------------------
# 5. Web pages
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Faster loading: compress text (page, data, scripts, styles) and let browsers
# keep the fonts / map library / scripts for a week instead of re-checking them.
# ---------------------------------------------------------------------------
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 7 * 24 * 3600
_COMPRESSIBLE = ("text/", "application/json", "application/javascript", "image/svg+xml")
_gz_cache = {}                                       # static files are compressed once, then reused


@app.after_request
def compress(resp):
    if (resp.status_code != 200 or "gzip" not in request.headers.get("Accept-Encoding", "")
            or resp.headers.get("Content-Encoding") or not resp.mimetype.startswith(_COMPRESSIBLE)):
        return resp
    key = None
    if resp.direct_passthrough:                      # a static file
        key = (request.path, resp.headers.get("ETag"))
        resp.direct_passthrough = False
    data = _gz_cache.get(key) if key else None
    if data is None:
        raw = resp.get_data()
        if len(raw) < 1024:
            return resp
        data = gzip.compress(raw, 6)
        if key:
            _gz_cache[key] = data
    resp.set_data(data)
    resp.headers["Content-Encoding"] = "gzip"
    resp.headers["Content-Length"] = str(len(data))
    resp.vary.add("Accept-Encoding")
    return resp

# Map style key: CARTO's free basemap key (carto.com/basemaps/apikey), saved in carto_key.txt.
# Without it, the map uses the standard OpenStreetMap style, which needs no key.
_key_file = os.path.join(HERE, "carto_key.txt")
CARTO_KEY = open(_key_file, encoding="utf-8").read().strip() if os.path.exists(_key_file) else ""

# Google Maps: a browser API key in maps_key.txt (restrict it to your website in Google Cloud).
# When present, the map uses Google Maps; if Google fails to load, the page falls back to the map above.
_gkey_file = os.path.join(HERE, "maps_key.txt")
MAPS_KEY = open(_gkey_file, encoding="utf-8").read().strip() if os.path.exists(_gkey_file) else ""
app.json.sort_keys = False          # keep the currency list in our order (USD, AED, SAR, ...)


def _sample_estimate():
    c = "Downtown Dubai"
    area = OPTIONS["typical_area"][c].get("2BR", 1300)
    # the same starting details as the Estimate form, so the hero search gives the same number
    form = {"community": c, "property_type": "2BR", "area_sqft": area, "floor": 10, "year_built": 2018,
            "parking_spaces": 2, "view": "community", "furnishing": "unfurnished", "condition": "vacant_on_transfer",
            "chiller_included": True}
    return {"community": c, "property_type": "2BR", "area_sqft": int(area),
            "price_usd": round(float(predict_price(build_row(form))[0])), "yoy_pct": COMMUNITY_YOY.get(c, 0.0)}


OPTIONS["sample"] = _sample_estimate()
# Price slider scale on the Find homes page: 40 log-spaced bands across recent sale prices
_lo, _hi = np.percentile(RECENT["price_usd"], [1, 99])
PRICE_EDGES = [float(v) for v in np.geomspace(_lo, _hi, 41)]
OPTIONS["price_edges"] = PRICE_EDGES
OPTIONS["seasons"] = SEASONS

# Price per sq ft by distance from Burj Khalifa (rings at 5, 10, 20 and 30 km), for the 3D city's
# Burj Khalifa view. Only used for display, never as a model input.
_burj_band = pd.cut(df["to_burj_khalifa_km"], [0, 5, 10, 20, 30, np.inf], right=False)
OPTIONS["burj_bands"] = [{"from": int(b.left), "to": None if np.isinf(b.right) else int(b.right),
                          "ppsf": int(g["price_per_sqft_usd"].median()), "sales": int(len(g))}
                         for b, g in df.groupby(_burj_band, observed=True)]


@app.get("/")
def home():
    r = _rates_cache                               # current rates, no network call here
    fx = {"rates": r["rates"], "updated": r["updated"], "live": r["live"]}
    return render_template("index.html", options=OPTIONS, fx=fx, carto_key=CARTO_KEY, maps_key=MAPS_KEY)


@app.post("/api/search")
def api_search():
    try:
        return jsonify(search_homes(request.get_json(force=True)))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Invalid search: {e}"), 400


@app.post("/api/sell")
def api_sell():
    try:
        return jsonify(sell_advice(request.get_json(force=True)))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Invalid or missing input: {e}"), 400


@app.get("/api/photos")
def api_photos():
    return jsonify(photos.snapshot())


@app.get("/api/metro")
def api_metro():
    return jsonify(metro.snapshot())


@app.get("/api/rates")
def api_rates():
    r = get_rates()
    return jsonify(rates=r["rates"], updated=r["updated"], live=r["live"])


@app.post("/api/predict")
def api_predict():
    form = request.get_json(force=True)
    try:
        X = build_row(form)
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Invalid or missing input: {e}"), 400

    price = float(predict_price(X)[0])
    sqft = float(form["area_sqft"])
    return jsonify(
        price_usd=round(price),
        price_per_sqft_usd=round(price / sqft),
        range_10=[round(price * 0.9), round(price * 1.1)],
        similar=similar_sales(form["community"], form["property_type"]),
        comparables=comparable_sales(form["community"], form["property_type"], sqft),
        trend=price_trend(form["community"]),
        what_if=what_if(form, price),
    )


if __name__ == "__main__":
    photos.start()                                  # looks up community photos in the background (cached after the first run)
    metro.start()                                   # downloads the metro lines once (cached in metro_cache.json)
    print("\n  Open http://127.0.0.1:5000 in your browser  (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
