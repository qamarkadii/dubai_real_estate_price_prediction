"""
Community photos for the Find homes cards.

The dataset has no property photos, so each card shows a real photo of the home's
community, taken from Wikipedia / Wikimedia Commons (free licences; the author and
licence are shown on every photo, as the licences require).

How it works
  * Each community has a short list of Wikipedia pages to try: its own page first,
    then a nearby, well-known place (e.g. Remraam -> Dubailand -> Global Village).
  * For each place we collect up to 24 photos: the Wikipedia page's main photo first, then
    Wikimedia Commons search results for that place (quality-reviewed photos first).
    Maps, logos, flags, portraits and small images are skipped. The page shows each photo
    at most once, so a community needs many photos to avoid repeats.
  * Only files hosted on Wikimedia Commons are used, so every photo is freely licensed.
  * This runs once in the background when the app starts and is saved to
    photos_cache.json, so later starts are instant. Delete that file to refresh.
"""

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "photos_cache_v2.json")      # v2: up to 24 photos per community
PER_PLACE = 24
UA = "DXBPrice/1.0 (https://dxbprice.me; university student project)"   # Wikimedia asks for a clear User-Agent
WIDTH = 960                                                                # thumbnail width served to the browser
SKIP = re.compile(r"map|logo|locator|location|flag|seal|emblem|coat|diagram|plan|icon|symbol|banner|signature|chart|"
                  r"portrait|selfie|people|person|\bman\b|woman|girl|boy|family|wedding|crowd|poster|screenshot|ticket|menu|sign\b", re.I)

# Wikipedia pages to try for each community, best first. The last entries are nearby landmarks.
_GOLF = ["Emirates Hills", "Emirates Golf Club"]
_DUBAILAND = ["Dubailand", "Global Village (Dubai)"]
_JUMEIRAH = ["Jumeirah", "Burj Al Arab", "Jumeirah Mosque"]
_MBR = ["Mohammed bin Rashid City", "Meydan Racecourse", "Dubai Hills Estate"]
CANDIDATES = {
    "Al Barsha": ["Al Barsha", "Mall of the Emirates"],
    "Al Barsha South": ["Al Barsha South", "Al Barsha", "Mall of the Emirates"],
    "Al Furjan": ["Al Furjan", "Ibn Battuta Mall"],
    "Al Mamzar": ["Al Mamzar", "Al Mamzar Beach Park"],
    "Al Nahda": ["Al Nahda, Dubai", "Al Qusais", "Deira"],
    "Al Qusais": ["Al Qusais", "Deira"],
    "Al Warqa": ["Al Warqa'a", "Mirdif"],
    "City Walk": ["City Walk (Dubai)", "Al Wasl", "Downtown Dubai"],
    "Al Reem": ["Arabian Ranches"] + _DUBAILAND,
    "Arabian Ranches": ["Arabian Ranches"] + _DUBAILAND,
    "Arabian Ranches 2": ["Arabian Ranches"] + _DUBAILAND,
    "Arabian Ranches 3": ["Arabian Ranches"] + _DUBAILAND,
    "Bukadra": ["Bu Kadra", "Ras Al Khor Wildlife Sanctuary", "Dubai Creek"],
    "Bur Dubai": ["Bur Dubai", "Al Fahidi Historical Neighbourhood", "Dubai Creek"],
    "Karama": ["Al Karama", "Bur Dubai"],
    "Business Bay": ["Business Bay", "Dubai Water Canal", "Downtown Dubai"],
    "Dubai Creek Harbour": ["Dubai Creek Harbour", "Dubai Creek", "Ras Al Khor Wildlife Sanctuary"],
    "DIFC": ["Dubai International Financial Centre", "Downtown Dubai"],
    "Dubai Investment Park": ["Dubai Investment Park", "Jebel Ali"],
    "Green Community": ["Dubai Investment Park", "Jebel Ali"],
    "DAMAC Lagoons": ["Damac Hills"] + _DUBAILAND,
    "Damac Hills": ["Damac Hills"] + _DUBAILAND,
    "Damac Hills 2": ["Damac Hills"] + _DUBAILAND,
    "Deira": ["Deira", "Dubai Creek"],
    "Dubai Islands": ["Dubai Islands", "Deira"],
    "Discovery Gardens": ["Discovery Gardens", "Ibn Battuta Mall"],
    "Downtown Dubai": ["Downtown Dubai", "Burj Khalifa", "Dubai Fountain"],
    "Bluewaters Island": ["Bluewaters Island", "Ain Dubai", "Dubai Marina"],
    "Dubai Harbour": ["Dubai Harbour", "Dubai Marina"],
    "Dubai Marina": ["Dubai Marina"],
    "Jumeirah Beach Residence": ["Jumeirah Beach Residence", "Dubai Marina"],
    "Dubai Silicon Oasis": ["Dubai Silicon Oasis", "Dubai"],
    "Dubai South": ["Dubai South", "Al Maktoum International Airport", "Expo City Dubai"],
    "Expo City": ["Expo City Dubai", "Expo 2020"],
    "Dubai Sports City": ["Dubai Sports City", "Dubai International Cricket Stadium"],
    "Dubai Studio City": ["Dubai Studio City"] + _DUBAILAND,
    "The Valley": ["Al Qudra Lakes"] + _DUBAILAND,
    "Dubailand": _DUBAILAND,
    "Falcon City of Wonders": ["Falcon City of Wonders"] + _DUBAILAND,
    "Living Legends": _DUBAILAND,
    "Liwan": _DUBAILAND,
    "Majan": _DUBAILAND,
    "Mudon": ["Mudon"] + _DUBAILAND,
    "Remraam": ["Remraam"] + _DUBAILAND,
    "Emirates Hills": _GOLF,
    "Jumeirah Islands": ["Jumeirah Islands"] + _GOLF,
    "Jumeirah Park": ["Jumeirah Park"] + _GOLF,
    "The Lakes": ["The Lakes, Dubai"] + _GOLF,
    "The Meadows": ["The Meadows, Dubai"] + _GOLF,
    "The Springs": ["The Springs, Dubai"] + _GOLF,
    "Dubai Festival City": ["Dubai Festival City", "Dubai Creek"],
    "Dubai Production City": ["Dubai Production City", "Dubai Sports City"],
    "International City": ["International City", "Dragon Mart"],
    "Jumeirah Golf Estates": ["Jumeirah Golf Estates", "Emirates Golf Club"],
    "Jumeirah Lake Towers": ["Jumeirah Lake Towers", "Almas Tower", "Dubai Marina"],
    "Jumeirah Village Circle": ["Jumeirah Village Circle", "Jumeirah Village", "Dubai Marina"],
    "Jumeirah Village Triangle": ["Jumeirah Village Triangle", "Jumeirah Village", "Dubai Marina"],
    "Jebel Ali": ["Jebel Ali", "Port of Jebel Ali"],
    "Jumeirah 1": _JUMEIRAH,
    "Jumeirah 2": _JUMEIRAH,
    "Jumeirah 3": _JUMEIRAH,
    "Jumeirah Bay Island": ["Jumeirah Bay Island", "Bulgari Resort Dubai"] + _JUMEIRAH,
    "Bulgari Resort": ["Bulgari Resort Dubai", "Jumeirah Bay Island"] + _JUMEIRAH,
    "La Mer": ["La Mer (Dubai)"] + _JUMEIRAH,
    "Madinat Jumeirah Living": ["Madinat Jumeirah", "Burj Al Arab"],
    "Pearl Jumeira": ["Pearl Jumeira"] + _JUMEIRAH,
    "Umm Suqeim": ["Umm Suqeim", "Burj Al Arab"],
    "Azizi Riviera": _MBR,
    "Sobha Hartland": ["Sobha Hartland"] + _MBR,
    "Maritime City": ["Dubai Maritime City", "Port Rashid"],
    "Meydan": ["Meydan Racecourse", "Meydan"],
    "Mirdif": ["Mirdif", "City Centre Mirdif"],
    "Dubai Hills Estate": ["Dubai Hills Estate", "Dubai Hills Mall"] + _MBR,
    "Mohammed Bin Rashid City": _MBR,
    "Motor City": ["Motor City, Dubai", "Dubai Autodrome"],
    "Town Square": ["Town Square Dubai"] + _DUBAILAND,
    "Palm Jumeirah": ["Palm Jumeirah", "Atlantis The Palm"],
    "Mira": ["Arabian Ranches"] + _DUBAILAND,
    "Reem Mira": ["Arabian Ranches"] + _DUBAILAND,
    "Sustainable City": ["The Sustainable City"] + _DUBAILAND,
    "The Greens": ["The Greens, Dubai"] + _GOLF,
    "The Views": ["The Greens, Dubai"] + _GOLF,
    "Tilal Al Ghaf": ["Tilal Al Ghaf", "Al Barsha South"],
    "World Islands": ["The World (archipelago)", "Palm Jumeirah"],
}
LAST_RESORT = ["Burj Khalifa", "Downtown Dubai"]

_state = {"ready": False, "photos": {}}


def _get(url, params):
    full = url + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(full, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _clean(html):
    """Commons gives the author as HTML; keep plain text."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()[:80]


def _info(files, place):
    """Licence, author and a 960 px thumbnail for Commons files; keeps big landscape photos only."""
    out = {}
    for k in range(0, len(files), 40):
        info = _get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "titles": "|".join("File:" + n for n in files[k:k + 40]),
            "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": WIDTH})
        for p in info.get("query", {}).get("pages", []):
            ii = (p.get("imageinfo") or [None])[0]
            photo = _keep(p, ii, place)
            if photo:
                out[p["title"].split(":", 1)[1].replace("_", " ")] = photo
    return out


def _keep(p, ii, place):
    if p.get("missing") or not ii:                       # not on Commons (e.g. a non-free local file): skip
        return None
    w, h = ii.get("width", 0), ii.get("height", 1)
    if w < 1200 or w < h * 1.25:                         # big enough and landscape
        return None
    if SKIP.search(p["title"]) or not re.search(r"\.(jpe?g|webp)$", p["title"], re.I):
        return None
    meta = ii.get("extmetadata", {})
    return {
        "url": ii.get("thumburl") or ii.get("url"),
        "page": ii.get("descriptionurl"),
        "artist": _clean(meta.get("Artist", {}).get("value")) or "Wikimedia Commons",
        "license": _clean(meta.get("LicenseShortName", {}).get("value")),
        "place": place,
    }


def _commons_search(place, limit):
    """Photos on Wikimedia Commons whose title/description mention the place (quality-reviewed ones first)."""
    name = re.sub(r"\s*\(.*\)$", "", place).replace(", Dubai", "").strip()
    term = f'"{name}"' + ("" if "dubai" in name.lower() else " Dubai")
    found = []
    for extra in (' incategory:"Quality_images"', ""):
        q = _get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "generator": "search", "gsrnamespace": 6, "gsrlimit": limit,
            "gsrsearch": term + " filetype:bitmap" + extra,
            "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": WIDTH})
        pages = sorted(q.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
        for p in pages:
            photo = _keep(p, (p.get("imageinfo") or [None])[0], place)
            if photo and all(photo["url"] != f["url"] for f in found):
                found.append(photo)
        if len(found) >= PER_PLACE:
            break
        time.sleep(0.1)
    return found


def _photos_for_place(title):
    """The page's own photos first, then Commons search results, up to PER_PLACE."""
    photos, place = _photos_on_page(title)
    try:
        for p in _commons_search(place or title, 50):
            if len(photos) >= PER_PLACE:
                break
            if all(p["url"] != q["url"] for q in photos):
                photos.append(p)
    except Exception as e:
        print(f"[photos] search {title}: {e}")
    return photos


def _photos_on_page(title):
    """Good, freely licensed landscape photos from one Wikipedia page (main photo first) and its title."""
    q = _get("https://en.wikipedia.org/w/api.php", {
        "action": "query", "redirects": 1, "titles": title,
        "prop": "pageimages|images", "piprop": "name", "imlimit": 60})
    pages = q.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return [], None
    page = pages[0]
    lead = page.get("pageimage")
    names = ([lead] if lead else []) + [i["title"].split(":", 1)[1] for i in page.get("images", [])]
    key = page["title"].split(",")[0].split("(")[0].strip().split()[-1].lower()   # e.g. "marina", "palm"
    picked = []
    for n in names:
        if n in picked or not re.search(r"\.(jpe?g|webp)$", n, re.I) or SKIP.search(n):
            continue
        if n != lead and key not in n.lower():          # extra photos must name the place, so they show it
            continue
        picked.append(n)
        if len(picked) == 30:
            break
    if not picked:
        return [], page["title"]
    by_name = _info(picked, page["title"])
    return [by_name[n.replace("_", " ")] for n in picked if n.replace("_", " ") in by_name], page["title"]


def _build():
    found, page_cache = {}, {}
    for community, titles in CANDIDATES.items():
        best = []
        for title in titles + LAST_RESORT:
            if title not in page_cache:
                try:
                    page_cache[title] = _photos_for_place(title)
                except Exception as e:                     # offline, blocked, timeout: try the next one
                    print(f"[photos] {title}: {e}")
                    page_cache[title] = []
                time.sleep(0.1)                            # be gentle with the Wikimedia servers
            if len(page_cache[title]) > len(best):
                best = page_cache[title]
            if len(best) >= 3:                             # enough photos of this place (or the nearest one)
                break
        if best:
            found[community] = best
        _state["photos"] = dict(found)                    # cards can use photos as soon as they arrive
    return found


def _worker():
    try:
        if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 30 * 86400:
            with open(CACHE, encoding="utf-8") as f:
                _state["photos"] = json.load(f)
        else:
            photos = _build()
            if photos:
                with open(CACHE, "w", encoding="utf-8") as f:
                    json.dump(photos, f, ensure_ascii=False, indent=1)
            print(f"[photos] found photos for {len(photos)} of {len(CANDIDATES)} communities")
    except Exception as e:
        print(f"[photos] could not load photos: {e}")
    _state["ready"] = True


def start():
    threading.Thread(target=_worker, daemon=True).start()


def snapshot():
    return {"ready": _state["ready"], "photos": _state["photos"]}
