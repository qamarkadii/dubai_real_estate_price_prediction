"""
Dubai Metro lines and stations for the "Metro" switch on the maps.

The dataset names each home's nearest metro station but has no station coordinates,
so the lines and stations are taken from OpenStreetMap (© OpenStreetMap contributors,
ODbL licence) through the free Overpass API.

How it works
  * One query when the app starts, in the background: every metro route in Dubai
    (Red Line, Green Line and the Route 2020 branch) with its track shape, plus every
    metro station with its English and Arabic name.
  * The result is saved to metro_cache.json, so later starts are instant and work offline.
    Delete that file to refresh.
  * If the laptop is offline the first time, the Metro switch is simply hidden.
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "metro_cache.json")
UA = "DXBPrice/1.0 (https://dxbprice.me; university student project)"
SERVERS = ["https://overpass-api.de/api/interpreter",          # tried in order
           "https://overpass.kumi.systems/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
BBOX = "24.75,54.85,25.45,55.65"                               # south, west, north, east: all of Dubai
QUERY = f"""
[out:json][timeout:90];
relation["route"="subway"]({BBOX});
out geom;
nwr["railway"="station"]({BBOX});
out center tags;
"""
RED, GREEN = "#E21836", "#00A651"                              # the official line colours

_state = {"ready": False, "lines": [], "stations": []}


def _fetch():
    body = urllib.parse.urlencode({"data": QUERY}).encode()
    last = None
    for url in SERVERS:
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:                                 # busy or blocked: try the next server
            last = e
    raise last


def _colour(tags):
    name = (tags.get("name:en") or tags.get("name") or "") + " " + tags.get("ref", "")
    if "green" in name.lower():
        return GREEN
    if "red" in name.lower() or "2020" in name:
        return RED
    c = tags.get("colour", "").strip()
    return c if c.startswith("#") else RED


def _is_metro(tags):
    """Metro stations only: not the Dubai Tram, the Palm Monorail or the Etihad Rail."""
    text = " ".join(tags.get(k, "") for k in ("network", "operator", "name:en", "name")).lower()
    if any(w in text for w in ("tram", "monorail", "etihad")):
        return False
    return tags.get("station") == "subway" or tags.get("subway") == "yes" or "metro" in text


def _parse(data):
    lines, stations, seen, used_ways = [], [], set(), set()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "relation" and tags.get("route") == "subway":
            if any(w in " ".join(tags.values()).lower() for w in ("tram", "monorail")):
                continue
            paths = []
            for m in el.get("members", []):
                # each route is mapped once per direction over the same tracks: draw every track piece once
                if m.get("type") == "way" and m.get("role", "") in ("", "forward", "backward") and m.get("geometry") \
                        and m.get("ref") not in used_ways:
                    used_ways.add(m.get("ref"))
                    paths.append([[round(p["lat"], 5), round(p["lon"], 5)] for p in m["geometry"]])
            if paths:
                lines.append({"name": tags.get("name:en") or tags.get("name", ""), "colour": _colour(tags), "paths": paths})
        elif tags.get("railway") == "station" and _is_metro(tags):
            lat = el.get("lat", el.get("center", {}).get("lat"))
            lon = el.get("lon", el.get("center", {}).get("lon"))
            name = tags.get("name:en") or tags.get("name")
            if lat is None or not name:
                continue
            key = name                                         # a station can be mapped as a point and an area, or once per line
            if key in seen:
                continue
            seen.add(key)
            stations.append({"name": name.replace(" Metro Station", "").replace(" Station", ""),
                             "name_ar": tags.get("name:ar", ""), "lat": round(lat, 5), "lon": round(lon, 5)})
    return {"lines": lines, "stations": stations}


def _worker():
    if os.path.exists(CACHE):
        try:
            with open(CACHE, encoding="utf-8") as f:
                _state.update(json.load(f), ready=True)
            return
        except (OSError, ValueError):
            pass
    for attempt in range(3):
        try:
            result = _parse(_fetch())
            if result["lines"] and result["stations"]:
                _state.update(result, ready=True)
                with open(CACHE, "w", encoding="utf-8") as f:
                    json.dump(result, f, separators=(",", ":"))
                print(f"  Metro map: {len(result['lines'])} lines, {len(result['stations'])} stations (saved to metro_cache.json)")
                return
        except Exception as e:
            print(f"  Metro map: could not download yet ({e.__class__.__name__}); trying again shortly")
        time.sleep(20 * (attempt + 1))
    _state["ready"] = True                                     # give up for this run: the switch stays hidden


def start():
    threading.Thread(target=_worker, daemon=True).start()


def snapshot():
    return dict(_state)
