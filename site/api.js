/* DXB Price: the Flask back end (app.py), running in the browser.
 *
 * On the laptop, the page called app.py at /api/predict, /api/search, /api/sell, /api/photos,
 * /api/metro and /api/rates. A static website has no Python server, so this file answers those
 * same calls in the browser, with the same logic as app.py:
 *   - api/data.json holds the trained network's weights, the fitted one-hot encoder and scaler,
 *     the lookup tables app.py builds at start-up, and the sales data used for comparisons.
 *   - The network (235 -> 64 -> 32 -> 1, sigmoid) is evaluated here exactly as Keras does,
 *     and exp() turns its log(price) back into USD.
 * The page itself (index.html) is unchanged: its fetch("/api/...") calls are answered below.
 */
(function () {
  "use strict";
  const realFetch = window.fetch.bind(window);
  let D = null, loading = null;

  // ---------------------------------------------------------------- data
  function prepare(d) {
    const R = d.rows, C = d.cats, n = R.price_usd.length, rows = new Array(n);
    for (let i = 0; i < n; i++) {
      rows[i] = {
        community: C.community[R.community[i]], property_type: C.property_type[R.property_type[i]],
        property_category: C.property_category[R.property_category[i]], bedrooms: R.bedrooms[i],
        area_sqft: R.area_sqft[i], floor: R.floor[i], year_built: R.year_built[i], view: C.view[R.view[i]],
        furnishing: C.furnishing[R.furnishing[i]], condition: C.condition[R.condition[i]],
        parking_spaces: R.parking_spaces[i], chiller_included: !!R.chiller_included[i],
        price_usd: R.price_usd[i], date_listed: R.date_listed[i],
      };
    }
    d.all = rows;
    d.recent = rows.filter(r => r.date_listed >= d.recent_from);          // "recent sales" = 2025 and 2026
    d.byType = {};
    for (const r of rows) (d.byType[r.community + "|" + r.property_type] ??= []).push(r);
    // index of every one-hot column, so preprocessing is a lookup
    d.model.catIndex = d.model.categories.map(cats => Object.fromEntries(cats.map((c, j) => [c, j])));
    delete d.rows;
    return d;
  }
  const load = () => (loading ??= realFetch("api/data.json").then(r => r.json()).then(d => (D = prepare(d))));

  // ---------------------------------------------------------------- helpers (Python behaviour)
  class InputError extends Error {}
  const req = (obj, key) => { if (!(key in obj) || obj[key] === undefined) throw new InputError(`'${key}'`); return obj[key]; };
  const lookup = (obj, key) => { if (!Object.prototype.hasOwnProperty.call(obj, key)) throw new InputError(`'${key}'`); return obj[key]; };
  const num = v => { const x = Number(v); if (v === null || v === "" || !Number.isFinite(x)) throw new InputError(`could not convert '${v}' to a number`); return x; };
  const int = v => Math.trunc(num(v));
  const pyRound = (x, nd = 0) => {                                 // Python round(): halves go to the even number
    const m = 10 ** nd, y = x * m, f = Math.floor(y);
    const r = y - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(y);
    return nd ? r / m : r;
  };
  const median = arr => {                                          // numpy / pandas median
    const s = Float64Array.from(arr).sort(), k = s.length >> 1;
    return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
  };

  // ---------------------------------------------------------------- the model
  const ACT = { sigmoid: x => 1 / (1 + Math.exp(-x)), relu: x => Math.max(0, x), tanh: Math.tanh, linear: x => x };

  function preprocess(row) {                       // ColumnTransformer: one-hot, then StandardScaler, then booleans
    const M = D.model, x = [];
    M.cat_cols.forEach((col, i) => {
      const onehot = new Array(M.categories[i].length).fill(0), j = M.catIndex[i][String(row[col])];
      if (j !== undefined) onehot[j] = 1;           // handle_unknown="ignore"
      x.push(...onehot);
    });
    M.num_cols.forEach((col, i) => x.push((Number(row[col]) - M.mean[i]) / M.scale[i]));
    M.bool_cols.forEach(col => x.push(row[col] ? 1 : 0));
    return x;
  }

  function predictPrice(row) {                     // Dense layers, then exp() of the predicted log(price)
    let a = preprocess(row);
    for (const L of D.model.layers) {
      const out = L.b.slice();
      for (let i = 0; i < a.length; i++) {
        if (a[i] === 0) continue;
        const Wi = L.W[i];
        for (let j = 0; j < out.length; j++) out[j] += a[i] * Wi[j];
      }
      a = out.map(ACT[L.act]);
    }
    return Math.exp(a[0]);
  }

  // ---------------------------------------------------------------- app.py, function by function
  function buildRow(form) {
    const community = req(form, "community"), c = lookup(D.communities, community);
    const ptype = req(form, "property_type");
    const category = ptype.endsWith("_villa") ? "villa" : "apartment";
    const listingMonth = form.listing_month || D.latest_month;
    const [year, month] = listingMonth.split("-").map(Number);
    const floor = num(form.floor || 0);
    const total = form.total_floors ? num(form.total_floors) : Math.max(D.typical_floors[community] ?? 40, floor);
    const station = form.metro_station || c.metro_station;
    const row = {
      community, zone: c.zone, is_freehold: c.is_freehold, lat: c.lat, lon: c.lon,
      property_category: category, property_type: ptype, bedrooms: lookup(D.bedrooms, ptype),
      area_sqft: num(req(form, "area_sqft")), floor, total_floors: total,
      year_built: int(req(form, "year_built")), view: req(form, "view"), furnishing: req(form, "furnishing"),
      condition: req(form, "condition"), parking_spaces: int(req(form, "parking_spaces")),
      chiller_included: !!form.chiller_included,
      metro_station: station, metro_line: lookup(D.metro_line, station),
      metro_distance_min: int(form.metro_distance_min || c.metro_distance_min),
      metro_distance_type: form.metro_distance_type || c.metro_distance_type,
      to_burj_khalifa_km: c.to_burj_khalifa_km,
      mortgage_rate_at_listing: lookup(D.mortgage_rate, listingMonth),
      listing_year: year, listing_month: month,
    };
    if (category === "villa") { row.floor = 0; row.total_floors = 0; }   // same rule as the notebook
    return row;
  }

  const similarSales = (community, ptype) => D.similar[community + "|" + ptype] ?? null;
  const priceTrend = community => lookup(D.trend, community);

  function comparableSales(community, ptype, area, n = 5) {
    const s = (D.byType[community + "|" + ptype] || []).map((r, i) => ({ r, i, gap: Math.abs(r.area_sqft - area) }));
    s.sort((a, b) => a.gap - b.gap || a.i - b.i);
    return s.slice(0, n).map(o => o.r)
      .sort((a, b) => (a.date_listed < b.date_listed ? 1 : a.date_listed > b.date_listed ? -1 : 0))
      .map(r => ({ date: r.date_listed.slice(0, 7), area_sqft: r.area_sqft, floor: r.floor,
                   view: r.view, furnishing: r.furnishing, price_usd: r.price_usd }));
  }

  function whatIf(form, base) {
    const variants = [];
    for (const [field, values] of [["view", D.views], ["furnishing", D.furnishing], ["condition", D.conditions]])
      for (const v of values) if (v !== form[field]) variants.push([field, v]);
    const p = int(form.parking_spaces);
    for (const x of [p - 1, p + 1]) if (x >= 1 && x <= 4) variants.push(["parking_spaces", String(x)]);
    const y = int(form.year_built);
    for (const x of [y - 5, y + 5]) if (x >= 2008 && x <= 2023) variants.push(["year_built", String(x)]);
    variants.push(["chiller_included", !form.chiller_included]);

    const best = new Map();
    for (const [field, value] of variants) {
      const delta = predictPrice(buildRow({ ...form, [field]: value })) - base;
      if (Math.abs(delta) < 0.005 * base) continue;                       // ignore changes under 0.5%
      const key = field + "|" + (delta > 0);
      if (!best.has(key) || Math.abs(delta) > Math.abs(best.get(key).delta_usd))
        best.set(key, { field, value, delta_usd: pyRound(delta), up: delta > 0 });
    }
    const all = [...best.values()];
    const ups = all.filter(v => v.up).sort((a, b) => b.delta_usd - a.delta_usd).slice(0, 3);
    const downs = all.filter(v => !v.up).sort((a, b) => a.delta_usd - b.delta_usd).slice(0, 2);
    return [...ups, ...downs].map(({ field, value, delta_usd }) => ({ field, value, delta_usd }));
  }

  function apiPredict(form) {
    const row = buildRow(form);
    const price = predictPrice(row), sqft = num(form.area_sqft);
    return {
      price_usd: pyRound(price), price_per_sqft_usd: pyRound(price / sqft),
      range_10: [pyRound(price * 0.9), pyRound(price * 1.1)],
      similar: similarSales(form.community, form.property_type),
      comparables: comparableSales(form.community, form.property_type, sqft),
      trend: priceTrend(form.community),
      what_if: whatIf(form, price),
    };
  }

  function groupBy(rows, key) {
    const g = new Map();
    for (const r of rows) { const k = r[key]; if (!g.has(k)) g.set(k, []); g.get(k).push(r); }
    return [...g.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));   // pandas sorts group keys
  }

  function histogram(values, edges) {
    // Same result as app.py's np.histogram(prices.clip(lo, hi), bins=edges). The prices are whole
    // numbers, so pandas stores the top bound rounded up (6,397,632 instead of 6,397,631.99...):
    // prices above it end just past the last edge and numpy leaves them out of the chart.
    const counts = new Array(edges.length - 1).fill(0), lo = edges[0], hi = edges[edges.length - 1];
    for (let v of values) {
      v = Math.max(v, lo);
      if (v > hi) continue;
      let a = 0, b = edges.length - 1;
      while (b - a > 1) { const m = (a + b) >> 1; if (v >= edges[m]) a = m; else b = m; }
      counts[Math.min(a, counts.length - 1)]++;
    }
    return counts;
  }

  function apiSearch(q) {
    let s = D.recent;
    if (q.category === "apartment" || q.category === "villa") s = s.filter(r => r.property_category === q.category);
    const bmin = int(q.bedrooms_min || 0);
    if (bmin > 0) s = s.filter(r => r.bedrooms >= bmin);
    const community = q.community;
    const inArea = community ? s.filter(r => r.community === community) : s;
    const hist = histogram(inArea.map(r => r.price_usd), D.price_edges);
    const max = num(q.budget_max_usd || 0), min = num(q.budget_min_usd || 0);
    if (max > 0) s = s.filter(r => r.price_usd <= max);
    if (min > 0) s = s.filter(r => r.price_usd >= min);
    const map = {};
    for (const [c, g] of groupBy(s, "community")) map[c] = [g.length, Math.trunc(median(g.map(r => r.price_usd)))];
    if (community) s = s.filter(r => r.community === community);

    const by = groupBy(s, "community")
      .map(([c, g]) => ({ community: c, count: g.length, median_price: Math.trunc(median(g.map(r => r.price_usd))),
                          median_area: Math.trunc(median(g.map(r => r.area_sqft))) }))
      .sort((a, b) => b.count - a.count).slice(0, 6);
    const order = { newest: ["date_listed", false], price_asc: ["price_usd", true],
                    price_desc: ["price_usd", false], size_desc: ["area_sqft", false] };
    const [col, asc] = order[q.sort] || order.newest;
    const page = int(q.page || 0), size = 12;
    const sorted = s.slice().sort((a, b) => {
      const d = a[col] < b[col] ? -1 : a[col] > b[col] ? 1 : 0;
      return asc ? d : -d;
    });
    return {
      total: s.length, hist, map, communities: by,
      homes: sorted.slice(page * size, (page + 1) * size).map(r => ({
        community: r.community, property_type: r.property_type, area_sqft: r.area_sqft, floor: r.floor,
        year_built: r.year_built, view: r.view, furnishing: r.furnishing, condition: r.condition,
        parking_spaces: r.parking_spaces, chiller_included: r.chiller_included, price_usd: r.price_usd,
        date: r.date_listed.slice(0, 7) })),
    };
  }

  function apiSell(form) {
    const value = predictPrice(buildRow(form));
    const community = form.community;
    const comps = comparableSales(community, form.property_type, num(form.area_sqft), 6);
    const asking = num(form.asking_price_usd || 0), bought = num(form.purchase_price_usd || 0);
    const boughtYear = int(form.purchase_year || 0);
    const yearsHeld = boughtYear ? Math.max(0.5, Number(D.latest_month.slice(0, 4)) - boughtYear + 0.5) : null;
    const yoy = D.yoy[community] ?? 0.0;
    const S = D.seasons, names = Object.keys(S);
    const bestSeason = names.reduce((b, k) => (S[k] > S[b] ? k : b), names[0]);
    const spread = Math.max(...Object.values(S)) - Math.min(...Object.values(S));
    const timing = yoy >= 5 ? "hold" : yoy <= -3 ? "now" : "season";
    return {
      value_usd: pyRound(value), range_10: [pyRound(value * 0.9), pyRound(value * 1.1)],
      comparables: comps,
      asking_usd: pyRound(asking) || null,
      asking_diff_pct: asking ? pyRound((asking / value - 1) * 100, 1) : null,
      profit_usd: bought ? pyRound(value - bought) : null,
      profit_pct: bought ? pyRound((value / bought - 1) * 100, 1) : null,
      annual_pct: bought && yearsHeld ? pyRound(((value / bought) ** (1 / yearsHeld) - 1) * 100, 1) : null,
      yoy_pct: yoy, timing, best_season: bestSeason, seasons: S, season_spread: pyRound(spread, 1),
    };
  }

  // Exchange rates: live from the same free API app.py used, or the fixed rates if it is unreachable
  let rates = null;
  async function apiRates() {
    if (rates) return rates;
    await load();
    try {
      const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), 4000);
      const data = await (await realFetch("https://open.er-api.com/v6/latest/USD", { signal: ctl.signal })).json();
      clearTimeout(t);
      const r = {};
      for (const c of D.currencies) if (c in data.rates) r[c] = data.rates[c];
      rates = { rates: r, updated: new Date(data.time_last_update_unix * 1000).toISOString().slice(0, 10), live: true };
    } catch (e) {
      rates = { rates: D.fallback_rates, updated: "2026-10-14", live: false };
    }
    return rates;
  }

  // ---------------------------------------------------------------- answer the page's fetch("/api/...") calls
  const json = (obj, status = 200) =>
    new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
  const HANDLERS = {
    "/api/predict": ["Invalid or missing input", apiPredict],
    "/api/search": ["Invalid search", apiSearch],
    "/api/sell": ["Invalid or missing input", apiSell],
  };

  window.fetch = async function (input, init) {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return realFetch(input, init);
    const path = url.pathname;
    if (path === "/api/photos") return realFetch("api/photos.json");
    if (path === "/api/metro") return realFetch("api/metro.json");
    if (path === "/api/rates") return json(await apiRates());
    const h = HANDLERS[path];
    if (!h) return json({ error: "Not found" }, 404);
    await load();
    let body;
    try { body = JSON.parse(init && init.body ? init.body : "{}"); } catch (e) { return json({ error: "Bad JSON" }, 400); }
    try {
      return json(h[1](body));
    } catch (e) {
      if (e instanceof InputError || e instanceof TypeError) return json({ error: `${h[0]}: ${e.message}` }, 400);
      throw e;
    }
  };

  // start downloading the data as soon as the page has loaded, so the first estimate is instant
  const warm = () => load().catch(() => { loading = null; });
  if (document.readyState === "complete") warm(); else window.addEventListener("load", warm);
  window.DXB_API = { load, apiPredict, apiSearch, apiSell, buildRow, predictPrice };   // for testing
})();
