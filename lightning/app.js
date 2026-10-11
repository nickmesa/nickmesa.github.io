/* Pre-Genesis Lightning Monitor: reads the JSON written by the pipeline and draws it.
   No build step and no libraries. Settings a site owner may want to change are at the top. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- settings
  var EMBED = window.GLM_EMBED || null;                      // data inlined into the page (preview builds)
  var QUERY = new URLSearchParams(location.search);
  var DEMO = EMBED ? !!EMBED.demo : QUERY.has('demo');       // ?demo=1 shows the synthetic examples
  var BASE = DEMO ? 'demo/' : (window.GLM_DATA_BASE || 'data/');   // the examples always come from demo/
  // Fixed ranges for the color scale and the area/energy axes, e.g.
  //   window.GLM_DOMAINS = { area: [60, 3000], energy: [0.5, 500], life: [5, 240] };
  // Left unset, each record uses its own 2nd-98th percentile range.
  var DOMAINS = window.GLM_DOMAINS || {};
  var MAP_HALF_KM = 260;
  var REFRESH_MS = 10 * 60 * 1000;

  var VARS = {
    area: { label: 'Mean group area', short: 'Area', unit: 'km²', fallback: [60, 3000] },
    energy: { label: 'Mean group energy', short: 'Energy', unit: 'fJ', fallback: [0.5, 500] },
    life: { label: 'Cell lifetime', short: 'Lifetime', unit: 'min', fallback: [5, 240] }
  };
  var WINDOWS = [[1, '1 h'], [3, '3 h'], [6, '6 h'], [12, '12 h'], [24, '24 h'], [0, 'All']];
  var BINS = [[6, '30 min'], [12, '1 h'], [36, '3 h']];
  // one-hue ramp, light to dark (steps 200-700 of the blue scale)
  var RAMP = ['#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
  var RAMP_DARK = ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'];

  // ------------------------------------------------------------------- state
  var st = {
    index: null, mode: 'realtime', id: null, S: null, coast: null,
    cursor: 1, win: 6, colorBy: 'area', bin: 12, land: false, playing: false, hover: -1, C: {}
  };
  var $ = function (id) { return document.getElementById(id); };
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  // ----------------------------------------------------------------- helpers
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function fmtTime(sec, withYear) {
    var d = new Date(sec * 1000);
    return d.getUTCDate() + ' ' + MONTHS[d.getUTCMonth()] + (withYear ? ' ' + d.getUTCFullYear() : '') + ' ' +
      pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + 'Z';
  }
  function fmtDay(sec) { var d = new Date(sec * 1000); return d.getUTCDate() + ' ' + MONTHS[d.getUTCMonth()] + ' ' + d.getUTCFullYear(); }
  function fmtInt(v) {
    if (v == null || !isFinite(v)) return '–';
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 1 : 2) + 'M';
    if (Math.abs(v) >= 1e4) return (v / 1e3).toFixed(1) + 'K';
    return Math.round(v).toLocaleString('en-US');
  }
  function fmtVal(v) {
    if (v == null || !isFinite(v)) return '–';
    var a = Math.abs(v);
    return a >= 100 ? Math.round(v).toLocaleString('en-US') : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  }
  function fmtLatLon(lat, lon) {
    if (lat == null || lon == null) return '–';
    return Math.abs(lat).toFixed(1) + '°' + (lat >= 0 ? 'N' : 'S') + ' ' + Math.abs(lon).toFixed(1) + '°' + (lon < 0 ? 'W' : 'E');
  }
  function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    var p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
  }
  function lowerBound(arr, v) {
    var lo = 0, hi = arr.length;
    while (lo < hi) { var m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
    return lo;
  }
  function nice125(v, up) {
    var e = Math.pow(10, Math.floor(Math.log10(v))), m = v / e, steps = [1, 2, 5, 10], i;
    if (up) { for (i = 0; i < 4; i++) if (steps[i] >= m - 1e-9) return steps[i] * e; }
    for (i = 3; i >= 0; i--) if (steps[i] <= m + 1e-9) return steps[i] * e;
    return e;
  }
  function logTicks(lo, hi) {
    var out = [], e = Math.pow(10, Math.floor(Math.log10(lo)));
    for (; e <= hi * 1.0001; e *= 10) [1, 2, 5].forEach(function (m) { var v = m * e; if (v >= lo * 0.9999 && v <= hi * 1.0001) out.push(v); });
    return out;
  }
  function thin(ticks, max) {
    if (ticks.length <= max) return ticks;
    var dec = ticks.filter(function (v) { return Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9; });
    if (dec.length >= 2 && dec.length <= max) return dec;
    return [ticks[0], ticks[ticks.length - 1]];
  }
  function tickText(v) { return v >= 1000 ? (v / 1000) + 'K' : String(v); }
  function el(tag, attrs, text) {
    var svg = /^(svg|g|rect|line|path|circle|text|polyline|defs|pattern|clipPath|title)$/.test(tag);
    var node = svg ? document.createElementNS('http://www.w3.org/2000/svg', tag) : document.createElement(tag);
    for (var k in attrs || {}) if (attrs[k] != null) node.setAttribute(k, attrs[k]);
    if (text != null) node.textContent = text;
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }
  function readColors() {
    var cs = getComputedStyle(document.documentElement), C = {};
    ['bg', 'panel', 'ink', 'ink-2', 'muted', 'grid', 'axis', 'data', 'data-wash', 'cursor', 'cursor-fill', 'land', 'coast', 'gap', 'line']
      .forEach(function (n) { C[n] = cs.getPropertyValue('--' + n).trim(); });
    var m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(C.panel) || [0, 'ff', 'ff', 'ff'];
    C.dark = (parseInt(m[1], 16) * 0.299 + parseInt(m[2], 16) * 0.587 + parseInt(m[3], 16) * 0.114) < 128;
    C.mono = cs.getPropertyValue('--mono').trim() || 'monospace';
    st.C = C;
  }
  function rampColor(t) {
    var r = st.C.dark ? RAMP_DARK : RAMP;
    return r[Math.max(0, Math.min(r.length - 1, Math.round(t * (r.length - 1))))];
  }
  function scaleLog(dom) { var a = Math.log(dom[0]), b = Math.log(dom[1]); return function (v) { return (Math.log(Math.max(v, 1e-9)) - a) / (b - a); }; }
  function colorOf(S, i) { var d = S.dom[st.colorBy]; return rampColor(Math.max(0, Math.min(1, scaleLog(d)(S[st.colorBy][i])))); }

  // -------------------------------------------------------------------- data
  function getJSON(path) {
    if (EMBED) {
      if (path === 'index.json') return Promise.resolve(EMBED.index);
      var id = path.replace(/^storms\//, '').replace(/\.json$/, '');
      return EMBED.storms[id] ? Promise.resolve(EMBED.storms[id]) : Promise.reject(new Error('missing ' + path));
    }
    return fetch(BASE + path, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error(path + ': ' + r.status);
      return r.json();
    });
  }
  function loadCoast() {
    var got = EMBED ? Promise.resolve(EMBED.coast) : fetch('coast.json').then(function (r) { return r.json(); });
    return got.then(function (c) {
      st.coast = c.rings.map(function (flat) {
        var a = Float32Array.from(flat), x0 = 999, x1 = -999, y0 = 999, y1 = -999;
        for (var i = 0; i < a.length; i += 2) { x0 = Math.min(x0, a[i]); x1 = Math.max(x1, a[i]); y0 = Math.min(y0, a[i + 1]); y1 = Math.max(y1, a[i + 1]); }
        return { a: a, x0: x0, x1: x1, y0: y0, y1: y1 };
      });
    }).catch(function () { st.coast = []; });
  }

  function prepare(p) {
    var f = p.features, s = p.steps;
    var S = {
      p: p, n: f.k.length, t0: s.t0, dt: s.dt, ns: s.n,
      k: Int32Array.from(f.k), x: Float32Array.from(f.x), y: Float32Array.from(f.y), lat: f.lat, lon: f.lon,
      g: Float32Array.from(f.n), area: Float32Array.from(f.area), energy: Float32Array.from(f.energy),
      life: Float32Array.from(f.life), land: Uint8Array.from(f.land), cell: Int32Array.from(f.cell), dom: {}
    };
    Object.keys(VARS).forEach(function (key) {
      if (DOMAINS[key]) { S.dom[key] = DOMAINS[key]; return; }
      var vals = Array.prototype.filter.call(S[key], function (v) { return v > 0; }).sort(function (a, b) { return a - b; });
      if (vals.length < 5) { S.dom[key] = VARS[key].fallback; return; }
      var lo = nice125(quantile(vals, 0.02), false), hi = nice125(quantile(vals, 0.98), true);
      S.dom[key] = hi > lo ? [lo, hi] : [lo, lo * 10];
    });
    // background density of the whole record for the area-energy panel
    S.axis = { area: [nice125(S.dom.area[0] / 1.5, false), nice125(S.dom.area[1] * 1.5, true)], energy: [nice125(S.dom.energy[0] / 1.5, false), nice125(S.dom.energy[1] * 1.5, true)] };
    S.genesis = p.genesis ? Date.parse(p.genesis.replace('Z', ':00Z')) / 1000 : null;
    return S;
  }
  function windowSteps(S) { return st.win ? Math.min(S.ns, st.win * 3600 / S.dt) : S.ns; }
  function windowRange(S) {     // features whose step lies in (cursor - window, cursor]
    return [lowerBound(S.k, st.cursor - windowSteps(S)), lowerBound(S.k, st.cursor)];
  }
  function shown(S, i) { return st.land || !S.land[i]; }
  function stepCentre(S, k) {
    k = Math.max(0, Math.min(S.ns - 1, k));
    return [S.p.steps.clat[k], S.p.steps.clon[k]];
  }
  function motion(S, k) {       // heading (deg) and speed (kt) from centres 3 h apart
    var h = Math.round(1.5 * 3600 / S.dt), a = stepCentre(S, k - h), b = stepCentre(S, k + h);
    var hours = (Math.min(S.ns - 1, k + h) - Math.max(0, k - h)) * S.dt / 3600;
    if (!hours || a[0] == null || b[0] == null) return null;
    var dy = (b[0] - a[0]) * 111.19, dx = (b[1] - a[1]) * 111.19 * Math.cos((a[0] + b[0]) / 2 * Math.PI / 180);
    var dist = Math.hypot(dx, dy);
    return { hdg: (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360, kt: dist / hours / 1.852, still: dist < 1 };
  }

  // ---------------------------------------------------------------- the list
  function chipFor(e) {
    if (e.status === 'active') return ['Active', 'chip live'];
    if (e.outcome === 'developed') return ['Developed', 'chip'];
    return ['Did not develop', 'chip'];
  }
  function listed() {
    var all = st.index ? st.index.storms : [];
    if (st.mode === 'realtime') return all.filter(function (e) { return e.status === 'active'; });
    var b = $('f-basin').value, o = $('f-outcome').value, y = $('f-year').value;
    return all.filter(function (e) {
      return e.status !== 'active' && (!b || e.basin === b) && (!o || e.outcome === o) && (!y || String(e.year) === y);
    });
  }
  function renderList() {
    var list = clear($('storm-list')), items = listed();
    $('archive-filters').hidden = st.mode !== 'archive';
    items.forEach(function (e) {
      var li = el('li'), b = el('button', { type: 'button', 'class': 'storm', 'aria-current': e.id === st.id ? 'true' : 'false' });
      var row = el('span', { 'class': 'row' }), chip = chipFor(e);
      row.appendChild(el('span', { 'class': 'name' }, e.label));
      row.appendChild(el('span', { 'class': chip[1] }, chip[0]));
      b.appendChild(row);
      var where = e.last ? fmtLatLon(e.last.lat, e.last.lon) + (e.last.vmax ? ' · ' + e.last.vmax + ' kt' : '') : '';
      b.appendChild(el('span', { 'class': 'meta' }, e.status === 'active' ? where : fmtDay(e.t_start) + ' to ' + fmtDay(e.t_end)));
      var peak = Math.max.apply(null, e.spark.concat([1]));
      var svg = el('svg', { viewBox: '0 0 72 26', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
      svg.appendChild(el('rect', { x: 0, y: 25.5, width: 72, height: 0.5, fill: 'var(--axis)' }));
      e.spark.forEach(function (v, i) {
        if (v > 0) { var h = Math.max(1, 24 * v / peak); svg.appendChild(el('rect', { x: i + 0.1, y: 25.5 - h, width: 0.8, height: h, fill: 'var(--data)' })); }
      });
      b.appendChild(svg);
      b.appendChild(el('span', { 'class': 'meta' }, fmtInt(e.n_groups) + ' groups in ' + fmtInt(e.n_features) + ' features'));
      b.title = 'Bars: lightning groups per hour over the last 72 h of the record';
      b.addEventListener('click', function () { select(e.id); });
      li.appendChild(b);
      list.appendChild(li);
    });
    var none = $('list-empty');
    none.hidden = items.length > 0;
    if (!items.length) {
      none.textContent = st.mode === 'realtime'
        ? 'No invests or pre-genesis systems are being tracked in the Atlantic or East Pacific right now. Past cases are under Archive.'
        : 'No archived cases match these filters.';
    }
  }
  function setMode(mode, keep) {
    st.mode = mode;
    $('tab-realtime').setAttribute('aria-selected', mode === 'realtime');
    $('tab-archive').setAttribute('aria-selected', mode === 'archive');
    renderList();
    var items = listed();
    if (!keep || !items.some(function (e) { return e.id === st.id; })) {
      if (items.length) select(items[0].id); else { st.id = null; st.S = null; $('detail').hidden = true; }
    }
  }

  // -------------------------------------------------------------- selection
  function select(id, keepView) {
    stopPlay();
    st.id = id;
    try { history.replaceState(null, '', '#' + id); } catch (err) { /* sandboxed frame */ }
    renderList();
    return getJSON('storms/' + id + '.json').then(function (p) {
      if (st.id !== id) return;
      var wasEnd = st.S && st.cursor >= st.S.ns;
      st.S = prepare(p);
      if (!keepView || wasEnd) st.cursor = st.S.ns;
      st.cursor = Math.max(1, Math.min(st.S.ns, st.cursor));
      st.hover = -1;
      $('detail').hidden = false;
      $('cursor').max = st.S.ns;
      renderHeadline();
      renderMethod();
      renderAll();
    }).catch(function (err) {
      $('detail').hidden = true;
      showEmpty('Could not load this record', String(err.message || err));
    });
  }

  function renderHeadline() {
    var S = st.S, p = S.p, tr = p.track, last = tr.t.length - 1;
    $('st-label').textContent = p.label;
    var chip = chipFor(p);
    $('st-chip').textContent = chip[0];
    $('st-chip').className = chip[1];
    $('st-note').hidden = !p.note;
    $('st-note').textContent = p.note || '';
    var facts = clear($('st-facts'));
    function fact(k, v) { var d = el('div'); d.appendChild(el('dt', null, k)); d.appendChild(el('dd', null, v)); facts.appendChild(d); }
    if (last >= 0) {
      fact(p.status === 'active' ? 'Latest fix' : 'Last fix in record', fmtTime(tr.t[last]));
      fact('Position', fmtLatLon(tr.lat[last], tr.lon[last]));
      fact('Intensity', (tr.vmax[last] != null ? tr.vmax[last] + ' kt' : '–') + (tr.mslp[last] != null ? ' · ' + tr.mslp[last] + ' hPa' : '') + ' · ' + tr.type[last]);
    }
    if (p.genesis) fact('Genesis (first TS)', fmtTime(S.genesis, true));
    fact('Lightning record', fmtTime(S.t0) + ' to ' + fmtTime(S.t0 + S.ns * S.dt));
    var sats = p.steps.sat.map(function (s) { return s[1]; }).filter(function (v, i, a) { return v && a.indexOf(v) === i; });
    fact('GLM', sats.length ? sats.map(function (s) { return 'GOES-' + s; }).join(', ') : '–');
    renderTrack();
  }

  function renderMethod() {
    var S = st.S, m = S.p.method, info = S.p.info, box = clear($('method'));
    box.appendChild(el('h3', null, 'How these numbers are made'));
    box.appendChild(el('p', null,
      'GLM groups with a good quality flag are counted on a ' + m.grid_res_deg + '° grid every ' + m.time_res_min +
      ' minutes. tobac outlines each connected patch of lightning as a feature and links features through time into cells. ' +
      'A feature’s mean group area and mean group energy are its summed group area and optical energy divided by its group count. ' +
      'Only features within ' + m.radius_km + ' km of the center are kept. The center is interpolated between NHC best-track fixes.'));
    box.appendChild(el('p', null,
      'Cells that come within ' + m.land_km + ' km of land at any time are left out unless “Include cells near land” is on' +
      (info.cells_near_land ? ' (' + info.cells_near_land + ' such cells in this record).' : '.') +
      (S.p.steps.extrap_from >= 0 ? ' After the latest fix the center is extrapolated with the recent motion, so the newest hours can shift when the next fix arrives.' : '') +
      (S.genesis ? ' Genesis here is the first tropical-storm record in the best track.' : '')));
  }

  // ------------------------------------------------------------------ tiles
  function renderTiles() {
    var S = st.S, r = windowRange(S), area = [], energy = [], groups = 0, count = 0, cells = {}, lives = [];
    for (var i = r[0]; i < r[1]; i++) {
      if (!shown(S, i)) continue;
      count++; groups += S.g[i]; area.push(S.area[i]); energy.push(S.energy[i]);
      if (!cells[S.cell[i]]) { cells[S.cell[i]] = 1; lives.push(S.life[i]); }
    }
    var sortNum = function (a, b) { return a - b; };
    area.sort(sortNum); energy.sort(sortNum); lives.sort(sortNum);
    var tiles = clear($('tiles'));
    function tile(k, v, u) {
      var d = el('div', { 'class': 'tile' });
      d.appendChild(el('div', { 'class': 'k' }, k));
      var val = el('div', { 'class': 'v' }, v);
      if (u && v !== '–') val.appendChild(el('span', { 'class': 'u' }, u));
      d.appendChild(val);
      tiles.appendChild(d);
    }
    tile('Lightning groups', fmtInt(groups));
    tile('Features', fmtInt(count));
    tile('Median group area', fmtVal(quantile(area, 0.5)), 'km²');
    tile('Median group energy', fmtVal(quantile(energy, 0.5)), 'fJ');
    tile('Median cell lifetime', fmtVal(quantile(lives, 0.5)), 'min');
    var end = S.t0 + st.cursor * S.dt, w = windowSteps(S) * S.dt;
    $('scope').textContent = 'From ' + fmtTime(end - w) + ' to ' + fmtTime(end) + ', within ' + S.p.method.radius_km + ' km of the center' +
      (st.land ? ', including cells near land.' : ', cells near land left out.') + ' Medians are across features.';
  }

  // -------------------------------------------------------------------- map
  function fitCanvas(cv) {
    var box = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    var w = Math.max(50, Math.round(box.width)), h = Math.max(50, Math.round(box.height));
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }
  function dotRadius(groups) { return Math.min(15, 2.2 + 0.4 * Math.sqrt(groups)); }

  function drawMap() {
    var S = st.S, C = st.C, f = fitCanvas($('map')), ctx = f.ctx, w = f.w, h = f.h;
    var m = { l: 34, r: 8, t: 8, b: 22 }, side = Math.min(w - m.l - m.r, h - m.t - m.b);
    var ox = m.l + (w - m.l - m.r - side) / 2, oy = m.t, s = side / (2 * MAP_HALF_KM);
    var X = function (km) { return ox + side / 2 + km * s; }, Y = function (km) { return oy + side / 2 - km * s; };
    st.mapGeom = { X: X, Y: Y, ox: ox, oy: oy, side: side };
    var c = stepCentre(S, st.cursor - 1), clat = c[0], clon = c[1];

    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, side, side); ctx.clip();
    ctx.fillStyle = C.panel; ctx.fillRect(ox, oy, side, side);
    // land, placed relative to the centre at the cursor time
    if (st.coast && clat != null) {
      var kx = 111.19 * Math.cos(clat * Math.PI / 180), dLat = MAP_HALF_KM / 111.19 + 0.2, dLon = MAP_HALF_KM / kx + 0.2;
      ctx.fillStyle = C.land; ctx.strokeStyle = C.coast; ctx.lineWidth = 1;
      st.coast.forEach(function (ring) {
        if (ring.x1 < clon - dLon || ring.x0 > clon + dLon || ring.y1 < clat - dLat || ring.y0 > clat + dLat) return;
        ctx.beginPath();
        for (var i = 0; i < ring.a.length; i += 2) {
          var px = X((ring.a[i] - clon) * kx), py = Y((ring.a[i + 1] - clat) * 111.19);
          if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        }
        ctx.closePath(); ctx.fill(); ctx.stroke();
      });
    }
    // range rings and axes
    ctx.lineWidth = 1; ctx.strokeStyle = C.grid;
    ctx.beginPath(); ctx.moveTo(X(-MAP_HALF_KM), Y(0)); ctx.lineTo(X(MAP_HALF_KM), Y(0)); ctx.moveTo(X(0), Y(-MAP_HALF_KM)); ctx.lineTo(X(0), Y(MAP_HALF_KM)); ctx.stroke();
    ctx.beginPath(); ctx.arc(X(0), Y(0), 100 * s, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1.25;
    ctx.beginPath(); ctx.arc(X(0), Y(0), S.p.method.radius_km * s, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = C.muted; ctx.font = '11px ' + C.mono; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    var a45 = Math.PI / 4;
    ctx.fillText(S.p.method.radius_km + ' km', X(S.p.method.radius_km * Math.cos(a45)) + 3, Y(S.p.method.radius_km * Math.sin(a45)) - 1);
    ctx.fillText('100', X(100 * Math.cos(a45)) + 3, Y(100 * Math.sin(a45)) - 1);

    // features: oldest first, fading with age inside the window
    var r = windowRange(S), wsteps = windowSteps(S), i, age, rad;
    for (i = r[0]; i < r[1]; i++) {
      var flagged = S.land[i];
      if (flagged && !st.land) continue;
      age = (st.cursor - 1 - S.k[i]) / Math.max(1, wsteps);
      rad = dotRadius(S.g[i]);
      ctx.globalAlpha = 0.28 + 0.67 * (1 - age);
      ctx.beginPath(); ctx.arc(X(S.x[i]), Y(S.y[i]), rad, 0, 2 * Math.PI);
      if (flagged) { ctx.strokeStyle = C.muted; ctx.lineWidth = 1.5; ctx.stroke(); }
      else { ctx.fillStyle = colorOf(S, i); ctx.fill(); ctx.strokeStyle = C.panel; ctx.lineWidth = 1; ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
    // centre and motion
    var mv = motion(S, st.cursor - 1);
    ctx.strokeStyle = C.ink; ctx.fillStyle = C.ink; ctx.lineWidth = 1.5;
    if (mv && !mv.still) {
      var len = 46, ang = mv.hdg * Math.PI / 180, ex = X(0) + len * Math.sin(ang), ey = Y(0) - len * Math.cos(ang);
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 8 * Math.sin(ang - 0.42), ey + 8 * Math.cos(ang - 0.42));
      ctx.lineTo(ex - 8 * Math.sin(ang + 0.42), ey + 8 * Math.cos(ang + 0.42));
      ctx.closePath(); ctx.fill();
    }
    ctx.beginPath(); ctx.moveTo(X(0) - 6, Y(0)); ctx.lineTo(X(0) + 6, Y(0)); ctx.moveTo(X(0), Y(0) - 6); ctx.lineTo(X(0), Y(0) + 6); ctx.stroke();
    if (st.hover >= 0 && st.hover >= r[0] && st.hover < r[1]) {
      ctx.strokeStyle = C.cursor; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(X(S.x[st.hover]), Y(S.y[st.hover]), dotRadius(S.g[st.hover]) + 3, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();
    // frame and tick labels
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
    ctx.fillStyle = C.muted; ctx.font = '11px ' + C.mono;
    [-200, -100, 0, 100, 200].forEach(function (v) {
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(v, X(v), oy + side + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(v, ox - 5, Y(v));
    });
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillStyle = C['ink-2'];
    ctx.fillText('km east / north of center', ox + side - 6, oy + side - 5);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(mv ? (mv.still ? 'stationary' : 'moving ' + pad(Math.round(mv.hdg / 5) * 5 % 360).padStart(3, '0') + '° at ' + Math.round(mv.kt) + ' kt') : '', ox + 7, oy + 6);

    $('map-sub').textContent = 'each dot is one feature in one 5-minute period; older dots are fainter';
  }

  function renderMapLegend() {
    var S = st.S, v = VARS[st.colorBy], d = S.dom[st.colorBy], box = clear($('map-legend'));
    var ramp = st.C.dark ? RAMP_DARK : RAMP;
    var item = el('span', { 'class': 'item' });
    item.appendChild(el('span', null, v.label + ' (' + v.unit + ')'));
    var rb = el('span', { 'class': 'rampbox' }), bar = el('span', { 'class': 'ramp' });
    bar.style.background = 'linear-gradient(to right,' + ramp.join(',') + ')';
    var ticks = el('span', { 'class': 'ticks' });
    [d[0], Math.sqrt(d[0] * d[1]), d[1]].forEach(function (t) { ticks.appendChild(el('span', null, t >= 100 ? fmtInt(nice125(t, false)) : String(+nice125(t, false).toPrecision(2)))); });
    rb.appendChild(bar); rb.appendChild(ticks); item.appendChild(rb); box.appendChild(item);
    var size = el('span', { 'class': 'item' });
    var svg = el('svg', { width: 86, height: 30, 'aria-hidden': 'true' }), x = 6;
    [10, 100, 1000].forEach(function (g) { var r = dotRadius(g); x += r; svg.appendChild(el('circle', { cx: x, cy: 15, r: r, fill: 'none', stroke: 'var(--ink-2)', 'stroke-width': 1 })); x += r + 7; });
    size.appendChild(svg); size.appendChild(el('span', null, '10, 100, 1,000 groups')); box.appendChild(size);
    if (st.land) {
      var ring = el('span', { 'class': 'item' }), s2 = el('svg', { width: 16, height: 16, 'aria-hidden': 'true' });
      s2.appendChild(el('circle', { cx: 8, cy: 8, r: 5, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.5 }));
      ring.appendChild(s2); ring.appendChild(el('span', null, 'cell near land')); box.appendChild(ring);
    }
  }

  // ------------------------------------------------------ area-energy panel
  function drawDist() {
    var S = st.S, C = st.C, f = fitCanvas($('dist')), ctx = f.ctx, w = f.w, h = f.h;
    var m = { l: 46, r: 10, t: 8, b: 36 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
    var dx = S.axis.area, dy = S.axis.energy, sx = scaleLog(dx), sy = scaleLog(dy);
    var X = function (v) { return m.l + sx(v) * pw; }, Y = function (v) { return m.t + (1 - sy(v)) * ph; };
    st.distGeom = { X: X, Y: Y };
    ctx.font = '11px ' + C.mono; ctx.lineWidth = 1;
    // grid and ticks
    var xt = logTicks(dx[0], dx[1]), yt = logTicks(dy[0], dy[1]);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted;
    xt.forEach(function (v) { ctx.beginPath(); ctx.moveTo(Math.round(X(v)) + 0.5, m.t); ctx.lineTo(Math.round(X(v)) + 0.5, m.t + ph); ctx.stroke(); });
    yt.forEach(function (v) { ctx.beginPath(); ctx.moveTo(m.l, Math.round(Y(v)) + 0.5); ctx.lineTo(m.l + pw, Math.round(Y(v)) + 0.5); ctx.stroke(); });
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    var xLab = pw < 330 ? thin(xt, 4) : xt, yLab = ph < 260 ? thin(yt, 5) : yt;
    xLab.forEach(function (v) { ctx.fillText(tickText(v), X(v), m.t + ph + 5); });
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    yLab.forEach(function (v) { ctx.fillText(tickText(v), m.l - 6, Y(v)); });
    ctx.fillStyle = C['ink-2']; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillText('mean group area (km²)', m.l + pw, h - 3);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('mean group energy (fJ)', m.l + 6, m.t + 4);
    ctx.strokeStyle = C.axis; ctx.strokeRect(m.l + 0.5, m.t + 0.5, pw - 1, ph - 1);

    ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
    // whole record as a neutral density
    var nb = 30, grid = new Float32Array(nb * nb), peak = 0, i, bx, by;
    for (i = 0; i < S.n; i++) {
      if (!shown(S, i)) continue;
      bx = Math.floor(sx(S.area[i]) * nb); by = Math.floor(sy(S.energy[i]) * nb);
      if (bx < 0 || bx >= nb || by < 0 || by >= nb) continue;
      peak = Math.max(peak, ++grid[by * nb + bx]);
    }
    ctx.fillStyle = C.muted;
    for (by = 0; by < nb; by++) for (bx = 0; bx < nb; bx++) {
      var v = grid[by * nb + bx];
      if (!v) continue;
      ctx.globalAlpha = 0.06 + 0.3 * Math.sqrt(v / peak);
      ctx.fillRect(m.l + bx / nb * pw, m.t + (1 - (by + 1) / nb) * ph, pw / nb + 0.5, ph / nb + 0.5);
    }
    // features in the window
    var r = windowRange(S), areas = [], energies = [];
    ctx.globalAlpha = 0.85; ctx.lineWidth = 1; ctx.strokeStyle = C.panel;
    for (i = r[0]; i < r[1]; i++) {
      if (!shown(S, i)) continue;
      areas.push(S.area[i]); energies.push(S.energy[i]);
      ctx.beginPath(); ctx.arc(X(S.area[i]), Y(S.energy[i]), 3.2, 0, 2 * Math.PI);
      ctx.fillStyle = colorOf(S, i); ctx.fill(); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (areas.length) {
      areas.sort(function (a, b) { return a - b; }); energies.sort(function (a, b) { return a - b; });
      var ma = quantile(areas, 0.5), me = quantile(energies, 0.5);
      ctx.strokeStyle = C.cursor; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(ma), m.t); ctx.lineTo(X(ma), m.t + ph); ctx.moveTo(m.l, Y(me)); ctx.lineTo(m.l + pw, Y(me)); ctx.stroke();
      ctx.fillStyle = C.ink; ctx.font = '500 11px ' + C.mono;
      var right = X(ma) < m.l + pw - 120;
      ctx.textAlign = right ? 'left' : 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('median ' + fmtVal(ma) + ' km²', X(ma) + (right ? 5 : -5), m.t + ph - 5);
      var top = Y(me) > m.t + 34;
      ctx.textAlign = 'right'; ctx.textBaseline = top ? 'bottom' : 'top';
      ctx.fillText('median ' + fmtVal(me) + ' fJ', m.l + pw - 6, Y(me) + (top ? -4 : 4));
    }
    if (st.hover >= r[0] && st.hover < r[1]) {
      ctx.strokeStyle = C.cursor; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(X(S.area[st.hover]), Y(S.energy[st.hover]), 6.5, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();
    $('dist-sub').textContent = 'log scales; ' + fmtInt(areas.length) + ' features in the window';
    var box = clear($('dist-legend'));
    var a = el('span', { 'class': 'item' }), s1 = el('svg', { width: 14, height: 14, 'aria-hidden': 'true' });
    s1.appendChild(el('circle', { cx: 7, cy: 7, r: 4, fill: 'var(--data)' }));
    a.appendChild(s1); a.appendChild(el('span', null, 'feature in the window, colored as on the map')); box.appendChild(a);
    var b = el('span', { 'class': 'item' }), s2 = el('svg', { width: 14, height: 14, 'aria-hidden': 'true' });
    s2.appendChild(el('rect', { x: 2, y: 2, width: 10, height: 10, fill: 'var(--muted)', opacity: 0.3 }));
    b.appendChild(s2); b.appendChild(el('span', null, 'all features in this record')); box.appendChild(b);
    var c = el('span', { 'class': 'item' }), s3 = el('svg', { width: 18, height: 14, 'aria-hidden': 'true' });
    s3.appendChild(el('line', { x1: 1, y1: 7, x2: 17, y2: 7, stroke: 'var(--cursor)', 'stroke-width': 1.5 }));
    c.appendChild(s3); c.appendChild(el('span', null, 'window medians')); box.appendChild(c);
  }

  // ------------------------------------------------------------ time series
  function binSeries(S) {
    var B = st.bin, base = Math.round(S.t0 / S.dt), b0 = Math.floor(base / B), nb = Math.floor((base + S.ns - 1) / B) - b0 + 1;
    var bins = [], i, b;
    for (b = 0; b < nb; b++) {
      var k0 = (b0 + b) * B - base;
      bins.push({ k0: k0, k1: k0 + B, groups: 0, feats: 0, area: [], energy: [], partial: k0 < 0 || k0 + B > S.ns, ok: 0, steps: 0, pending: 0 });
    }
    for (i = 0; i < S.n; i++) {
      if (!shown(S, i)) continue;
      b = bins[Math.floor((base + S.k[i]) / B) - b0];
      b.groups += S.g[i]; b.feats++; b.area.push(S.area[i]); b.energy.push(S.energy[i]);
    }
    var nf = S.p.steps.nfiles, need = 0.8 * S.p.steps.expected_files;
    for (i = 0; i < S.ns; i++) {
      b = bins[Math.floor((base + i) / B) - b0];
      b.steps++; if (nf[i] >= need) b.ok++; else if (nf[i] < 0) b.pending++;
    }
    bins.forEach(function (bin) {
      ['area', 'energy'].forEach(function (key) {
        var a = bin[key].sort(function (x, y) { return x - y; });
        bin[key + 'Q'] = a.length ? [quantile(a, 0.25), quantile(a, 0.5), quantile(a, 0.75)] : null;
      });
      bin.cover = bin.steps ? bin.ok / bin.steps : 0;
    });
    return bins;
  }

  var TS_ROWS = [
    { key: 'groups', title: 'Lightning groups per bin', kind: 'bar' },
    { key: 'feats', title: 'Features per bin', kind: 'bar' },
    { key: 'area', title: 'Group area (km²): median and middle 50%', kind: 'band' },
    { key: 'energy', title: 'Group energy (fJ): median and middle 50%', kind: 'band' }
  ];

  function renderTS() {
    var S = st.S, svg = clear($('ts')), W = Math.max(300, Math.round($('ts-wrap').getBoundingClientRect().width));
    var m = { l: 46, r: 14, t: S.genesis ? 26 : 10 }, rowH = 78, gap = 26, pw = W - m.l - m.r;
    var bins = binSeries(S), tA = S.t0, tB = S.t0 + S.ns * S.dt;
    var X = function (t) { return m.l + (t - tA) / (tB - tA) * pw; }, XK = function (k) { return X(S.t0 + k * S.dt); };
    var qualityY = m.t + TS_ROWS.length * (rowH + gap) + 12, H = qualityY + 46;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('height', H);
    st.tsGeom = { X: X, XK: XK, m: m, pw: pw, bins: bins, top: m.t, bottom: qualityY + 8, W: W, H: H };

    var defs = el('defs'), pat = el('pattern', { id: 'pend', width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    pat.appendChild(el('rect', { width: 5, height: 5, fill: 'var(--panel)' }));
    pat.appendChild(el('rect', { width: 2, height: 5, fill: 'var(--gap)' }));
    defs.appendChild(pat); svg.appendChild(defs);

    svg.appendChild(el('g', { id: 'ts-window' }));                // window band sits behind the marks

    // time ticks
    var span = (tB - tA) / 3600, stepH = span > 150 ? 24 : (span > 60 || pw < 520) ? (pw < 420 && span > 60 ? 24 : 12) : 6, ticks = [], t;
    for (t = Math.ceil(tA / (stepH * 3600)) * stepH * 3600; t <= tB; t += stepH * 3600) ticks.push(t);

    TS_ROWS.forEach(function (row, ri) {
      var y0 = m.t + ri * (rowH + gap) + 16, y1 = y0 + rowH, g = el('g');
      g.appendChild(el('text', { x: m.l, y: y0 - 6, fill: 'var(--ink-2)', 'font-size': 11.5, style: 'font-family:var(--sans)' }, row.title));
      ticks.forEach(function (tk) { g.appendChild(el('line', { x1: X(tk), x2: X(tk), y1: y0, y2: y1, stroke: 'var(--grid)' })); });
      var yOf, yTicks;
      if (row.kind === 'bar') {
        var max = Math.max.apply(null, bins.map(function (b) { return b[row.key]; }).concat([1])), top = nice125(max, true);
        yOf = function (v) { return y1 - v / top * rowH; };
        yTicks = [0, top / 2, top];
      } else {
        var d = S.axis[row.key], sl = scaleLog(d);
        yOf = function (v) { return y1 - Math.max(0, Math.min(1, sl(v))) * rowH; };
        yTicks = thin(logTicks(d[0], d[1]), 4);
      }
      yTicks.forEach(function (v) {
        g.appendChild(el('line', { x1: m.l, x2: m.l + pw, y1: yOf(v), y2: yOf(v), stroke: v === 0 ? 'var(--axis)' : 'var(--grid)' }));
        g.appendChild(el('text', { x: m.l - 6, y: yOf(v) + 3.5, 'text-anchor': 'end', fill: 'var(--muted)', 'font-size': 10.5 }, tickText(v)));
      });
      if (row.kind === 'bar') {
        var bw = XK(bins[0].k1) - XK(bins[0].k0), barW = Math.max(1, Math.min(24, bw - (bw > 5 ? 2 : 0.6)));
        bins.forEach(function (b) {
          var v = b[row.key]; if (!v) return;
          var xa = Math.max(m.l, XK(Math.max(0, b.k0))), xb = Math.min(m.l + pw, XK(Math.min(S.ns, b.k1)));
          var cx = (XK(b.k0) + XK(b.k1)) / 2, wBar = Math.min(barW, Math.max(1, xb - xa - 0.6)), x = Math.max(xa, Math.min(xb - wBar, cx - wBar / 2));
          var yy = yOf(v), hh = Math.max(1, y1 - yy), rr = Math.min(3, wBar / 2, hh);
          g.appendChild(el('path', {
            d: 'M' + x + ',' + y1 + 'V' + (yy + rr) + 'Q' + x + ',' + yy + ' ' + (x + rr) + ',' + yy + 'H' + (x + wBar - rr) + 'Q' + (x + wBar) + ',' + yy + ' ' + (x + wBar) + ',' + (yy + rr) + 'V' + y1 + 'Z',
            fill: 'var(--data)', opacity: b.partial ? 0.45 : 1
          }));
        });
      } else {
        var q = row.key + 'Q', seg = [], segs = [];
        bins.forEach(function (b) { if (b[q] && b.feats >= 3) seg.push(b); else if (seg.length) { segs.push(seg); seg = []; } });
        if (seg.length) segs.push(seg);
        var mid = function (b) { return Math.max(m.l, Math.min(m.l + pw, (XK(b.k0) + XK(b.k1)) / 2)); };
        segs.forEach(function (sg) {
          if (sg.length > 1) {
            var up = sg.map(function (b) { return mid(b) + ',' + yOf(b[q][2]); }), dn = sg.slice().reverse().map(function (b) { return mid(b) + ',' + yOf(b[q][0]); });
            g.appendChild(el('path', { d: 'M' + up.join('L') + 'L' + dn.join('L') + 'Z', fill: 'var(--data-wash)' }));
            g.appendChild(el('path', { d: 'M' + sg.map(function (b) { return mid(b) + ',' + yOf(b[q][1]); }).join('L'), fill: 'none', stroke: 'var(--data)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
          } else {
            g.appendChild(el('line', { x1: mid(sg[0]), x2: mid(sg[0]), y1: yOf(sg[0][q][2]), y2: yOf(sg[0][q][0]), stroke: 'var(--data-wash)', 'stroke-width': 5 }));
            g.appendChild(el('circle', { cx: mid(sg[0]), cy: yOf(sg[0][q][1]), r: 3, fill: 'var(--data)', stroke: 'var(--panel)', 'stroke-width': 1.5 }));
          }
        });
      }
      svg.appendChild(g);
    });

    // GLM coverage strip and time axis
    var q = el('g');
    q.appendChild(el('text', { x: m.l, y: qualityY - 4, fill: 'var(--ink-2)', 'font-size': 11.5, style: 'font-family:var(--sans)' }, 'GLM files received'));
    var nf = S.p.steps.nfiles, need = 0.8 * S.p.steps.expected_files, runStart = 0, kind = function (v) { return v >= need ? 'ok' : v < 0 ? 'pend' : 'gap'; };
    for (var k = 1; k <= S.ns; k++) {
      if (k === S.ns || kind(nf[k]) !== kind(nf[runStart])) {
        var kd = kind(nf[runStart]);
        q.appendChild(el('rect', { x: XK(runStart), y: qualityY, width: Math.max(0.5, XK(k) - XK(runStart)), height: 7, fill: kd === 'ok' ? 'var(--data)' : kd === 'pend' ? 'url(#pend)' : 'var(--gap)', opacity: kd === 'ok' ? 0.55 : 1 }));
        runStart = k;
      }
    }
    var legendX = m.l + 128;
    [['var(--data)', 'complete', 0.55], ['var(--gap)', 'missing', 1], ['url(#pend)', 'not downloaded yet', 1]].forEach(function (it) {
      if (legendX + 120 > W) return;
      q.appendChild(el('rect', { x: legendX, y: qualityY - 12, width: 9, height: 7, fill: it[0], opacity: it[2] }));
      q.appendChild(el('text', { x: legendX + 13, y: qualityY - 5, fill: 'var(--muted)', 'font-size': 10.5, style: 'font-family:var(--sans)' }, it[1]));
      legendX += 24 + it[1].length * 5.6;
    });
    var ex = S.p.steps.extrap_from;
    if (ex >= 0) {
      q.appendChild(el('line', { x1: XK(ex), x2: XK(S.ns), y1: qualityY + 10.5, y2: qualityY + 10.5, stroke: 'var(--ink-2)', 'stroke-width': 2 }));
      q.appendChild(el('text', { x: XK(S.ns), y: qualityY + 36, 'text-anchor': 'end', fill: 'var(--ink-2)', 'font-size': 10.5, style: 'font-family:var(--sans)' }, 'underlined: center extrapolated past the latest fix'));
    }
    var lastLabel = -99;
    ticks.forEach(function (tk) {
      var d = new Date(tk * 1000), hr = d.getUTCHours(), x = X(tk);
      q.appendChild(el('line', { x1: x, x2: x, y1: qualityY + 7, y2: qualityY + 11, stroke: 'var(--axis)' }));
      if (x - lastLabel < 46) return;
      var label = hr === 0 ? d.getUTCDate() + ' ' + MONTHS[d.getUTCMonth()] : pad(hr) + 'Z';
      q.appendChild(el('text', { x: x, y: qualityY + 23, 'text-anchor': x > m.l + pw - 24 ? 'end' : x < m.l + 20 ? 'start' : 'middle', fill: hr === 0 ? 'var(--ink-2)' : 'var(--muted)', 'font-size': 10.5 }, label));
      lastLabel = x;
    });
    svg.appendChild(q);
    if (S.genesis) {
      var ga = el('g');
      for (var hb = 0; hb <= 240; hb += (span > 100 || pw < 520 ? 24 : 12)) {
        var tg = S.genesis - hb * 3600;
        if (tg < tA) break;
        if (tg > tB) continue;
        ga.appendChild(el('line', { x1: X(tg), x2: X(tg), y1: 14, y2: 18, stroke: 'var(--axis)' }));
        ga.appendChild(el('text', { x: X(tg), y: 11, 'text-anchor': hb === 0 ? 'end' : 'middle', fill: 'var(--ink-2)', 'font-size': 10.5 }, hb === 0 ? 'genesis' : '−' + hb + ' h'));
      }
      svg.appendChild(ga);
    }
    svg.appendChild(el('g', { id: 'ts-hover' }));
    $('ts-sub').textContent = BINS.filter(function (b) { return b[0] === st.bin; })[0][1] + ' bins, UTC; shaded band is the selected window; click or drag to move it';
    renderTSCursor();
    renderTable(bins);
  }

  function renderTSCursor() {
    var G = st.tsGeom, S = st.S, g = $('ts-window');
    if (!G || !g) return;
    clear(g);
    var x1 = G.XK(st.cursor), x0 = G.XK(Math.max(0, st.cursor - windowSteps(S)));
    g.appendChild(el('rect', { x: x0, y: G.top, width: Math.max(1, x1 - x0), height: G.bottom - G.top, fill: 'var(--cursor-fill)' }));
    g.appendChild(el('line', { x1: x1, x2: x1, y1: G.top, y2: G.bottom, stroke: 'var(--cursor)', 'stroke-width': 1.5 }));
  }

  function renderTable(bins) {
    var S = st.S, table = clear($('ts-table')), head = el('tr');
    ['Bin start (UTC)', 'Groups', 'Features', 'Area p25', 'Area median', 'Area p75', 'Energy p25', 'Energy median', 'Energy p75', 'GLM coverage'].forEach(function (h) { head.appendChild(el('th', null, h)); });
    table.appendChild(head);
    bins.forEach(function (b) {
      var tr = el('tr'), a = b.areaQ || [NaN, NaN, NaN], e = b.energyQ || [NaN, NaN, NaN];
      [fmtTime(S.t0 + b.k0 * S.dt, true), fmtInt(b.groups), fmtInt(b.feats), fmtVal(a[0]), fmtVal(a[1]), fmtVal(a[2]), fmtVal(e[0]), fmtVal(e[1]), fmtVal(e[2]), Math.round(b.cover * 100) + '%']
        .forEach(function (v) { tr.appendChild(el('td', null, v)); });
      table.appendChild(tr);
    });
  }

  // ------------------------------------------------------------------ track
  function renderTrack() {
    var S = st.S, tr = S.p.track, svg = clear($('track-map')), W = 300, H = 168;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var lats = S.p.steps.clat.concat(tr.lat), lons = S.p.steps.clon.concat(tr.lon);
    lats = lats.filter(function (v) { return v != null; }); lons = lons.filter(function (v) { return v != null; });
    if (!lats.length) return;
    var la0 = Math.min.apply(null, lats), la1 = Math.max.apply(null, lats), lo0 = Math.min.apply(null, lons), lo1 = Math.max.apply(null, lons);
    var midLat = (la0 + la1) / 2, kx = Math.cos(midLat * Math.PI / 180);
    var spanLat = Math.max(la1 - la0 + 9, (lo1 - lo0 + 12) * kx * H / W), spanLon = spanLat * W / H / kx;
    var cLat = midLat, cLon = (lo0 + lo1) / 2, sc = H / spanLat;
    var X = function (lon) { return W / 2 + (lon - cLon) * kx * sc; }, Y = function (lat) { return H / 2 - (lat - cLat) * sc; };
    st.trackGeom = { X: X, Y: Y, sc: sc };
    var land = '';
    (st.coast || []).forEach(function (ring) {
      if (ring.x1 < cLon - spanLon / 2 - 1 || ring.x0 > cLon + spanLon / 2 + 1 || ring.y1 < cLat - spanLat / 2 - 1 || ring.y0 > cLat + spanLat / 2 + 1) return;
      for (var i = 0; i < ring.a.length; i += 2) land += (i ? 'L' : 'M') + X(ring.a[i]).toFixed(1) + ',' + Y(ring.a[i + 1]).toFixed(1);
      land += 'Z';
    });
    if (land) svg.appendChild(el('path', { d: land, fill: 'var(--land)', stroke: 'var(--coast)', 'stroke-width': 0.7 }));
    var stepDeg = spanLon > 40 ? 10 : 5, v;
    for (v = Math.ceil((cLon - spanLon / 2) / stepDeg) * stepDeg; v < cLon + spanLon / 2; v += stepDeg) {
      svg.appendChild(el('line', { x1: X(v), x2: X(v), y1: 0, y2: H, stroke: 'var(--grid)', 'stroke-width': 0.7 }));
      svg.appendChild(el('text', { x: X(v) + 2, y: H - 3, fill: 'var(--muted)', 'font-size': 8.5 }, Math.abs(v) + '°' + (v < 0 ? 'W' : 'E')));
    }
    for (v = Math.ceil((cLat - spanLat / 2) / stepDeg) * stepDeg; v < cLat + spanLat / 2; v += stepDeg) {
      svg.appendChild(el('line', { x1: 0, x2: W, y1: Y(v), y2: Y(v), stroke: 'var(--grid)', 'stroke-width': 0.7 }));
      svg.appendChild(el('text', { x: 3, y: Y(v) - 2, fill: 'var(--muted)', 'font-size': 8.5 }, Math.abs(v) + '°' + (v < 0 ? 'S' : 'N')));
    }
    var pts = [], stepN = Math.max(1, Math.round(S.ns / 200));
    for (var k = 0; k < S.ns; k += stepN) pts.push(X(S.p.steps.clon[k]).toFixed(1) + ',' + Y(S.p.steps.clat[k]).toFixed(1));
    pts.push(X(S.p.steps.clon[S.ns - 1]).toFixed(1) + ',' + Y(S.p.steps.clat[S.ns - 1]).toFixed(1));
    svg.appendChild(el('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.6, 'stroke-linejoin': 'round' }));
    tr.t.forEach(function (t, i) {
      svg.appendChild(el('circle', { cx: X(tr.lon[i]), cy: Y(tr.lat[i]), r: 2.3, fill: 'var(--panel)', stroke: 'var(--ink)', 'stroke-width': 1 }));
    });
    var g = el('g', { id: 'track-now' });
    g.appendChild(el('circle', { r: S.p.method.radius_km / 111.19 * sc, fill: 'var(--cursor-fill)', stroke: 'var(--cursor)', 'stroke-width': 1 }));
    g.appendChild(el('circle', { r: 3, fill: 'var(--cursor)', stroke: 'var(--panel)', 'stroke-width': 1.2 }));
    svg.appendChild(g);
    svg.appendChild(el('title', null, 'Track from NHC best-track fixes (open circles). The shaded circle is the 200 km analysis area at the selected time.'));
    moveTrackMarker();
  }
  function moveTrackMarker() {
    var g = $('track-now'), G = st.trackGeom;
    if (!g || !G) return;
    var c = stepCentre(st.S, st.cursor - 1);
    g.setAttribute('transform', 'translate(' + G.X(c[1]).toFixed(1) + ',' + G.Y(c[0]).toFixed(1) + ')');
  }

  // -------------------------------------------------------------- tooltips
  function showTip(evt, title, rows) {
    var tip = clear($('tip'));
    tip.appendChild(el('span', { 'class': 't' }, title));
    rows.forEach(function (r) { var d = el('div', { 'class': 'r' }); d.appendChild(el('span', null, r[0])); d.appendChild(el('b', null, r[1])); tip.appendChild(d); });
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight, x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - 12;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - 12;
    tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
  }
  function hideTip() { $('tip').hidden = true; }
  function featureTip(evt, i) {
    var S = st.S, rng = Math.hypot(S.x[i], S.y[i]), brg = (Math.atan2(S.x[i], S.y[i]) * 180 / Math.PI + 360) % 360;
    showTip(evt, fmtTime(S.t0 + S.k[i] * S.dt) + ' · ' + fmtLatLon(S.lat[i], S.lon[i]), [
      ['Groups', fmtInt(S.g[i])], ['Mean group area', fmtVal(S.area[i]) + ' km²'], ['Mean group energy', fmtVal(S.energy[i]) + ' fJ'],
      ['Cell lifetime', fmtInt(S.life[i]) + ' min'], ['From center', Math.round(rng) + ' km at ' + Math.round(brg) + '°']
    ].concat(S.land[i] ? [['Cell', 'near land']] : []));
  }
  function nearest(evt, cv, px, py, reach) {
    var S = st.S, box = cv.getBoundingClientRect(), mx = evt.clientX - box.left, my = evt.clientY - box.top, r = windowRange(S), best = -1, bd = 1e9;
    for (var i = r[0]; i < r[1]; i++) {
      if (!shown(S, i)) continue;
      var d = Math.hypot(px(i) - mx, py(i) - my) - reach(i);
      if (d < bd) { bd = d; best = i; }
    }
    return bd <= 12 ? best : -1;
  }
  function hoverCanvas(evt, which) {
    var S = st.S, G = which === 'map' ? st.mapGeom : st.distGeom;
    if (!S || !G) return;
    var i = which === 'map'
      ? nearest(evt, $('map'), function (j) { return G.X(S.x[j]); }, function (j) { return G.Y(S.y[j]); }, function (j) { return dotRadius(S.g[j]); })
      : nearest(evt, $('dist'), function (j) { return G.X(S.area[j]); }, function (j) { return G.Y(S.energy[j]); }, function () { return 3; });
    if (i !== st.hover) { st.hover = i; drawMap(); drawDist(); }
    if (i >= 0) featureTip(evt, i); else hideTip();
  }
  function tsPointer(evt, setCursor) {
    var S = st.S, G = st.tsGeom;
    if (!S || !G) return;
    var box = $('ts').getBoundingClientRect(), x = (evt.clientX - box.left) * G.W / box.width;
    var k = Math.max(0, Math.min(S.ns, (x - G.m.l) / G.pw * S.ns));
    if (setCursor) { setCursorStep(Math.max(1, Math.round(k))); }
    var bin = null;
    G.bins.forEach(function (b) { if (k >= b.k0 && k < b.k1) bin = b; });
    var hv = clear($('ts-hover'));
    if (!bin || x < G.m.l || x > G.m.l + G.pw) { hideTip(); return; }
    var cx = Math.max(G.m.l, Math.min(G.m.l + G.pw, (G.XK(bin.k0) + G.XK(bin.k1)) / 2));
    hv.appendChild(el('line', { x1: cx, x2: cx, y1: G.top, y2: G.bottom, stroke: 'var(--ink-2)', 'stroke-width': 1 }));
    var a = bin.areaQ, e = bin.energyQ, t0 = S.t0 + bin.k0 * S.dt, rows = [
      ['Groups', fmtInt(bin.groups)], ['Features', fmtInt(bin.feats)],
      ['Area', a ? fmtVal(a[1]) + ' km² (' + fmtVal(a[0]) + '–' + fmtVal(a[2]) + ')' : '–'],
      ['Energy', e ? fmtVal(e[1]) + ' fJ (' + fmtVal(e[0]) + '–' + fmtVal(e[2]) + ')' : '–'],
      ['GLM coverage', Math.round(bin.cover * 100) + '%' + (bin.partial ? ', partial bin' : '')]
    ];
    if (S.genesis) rows.unshift(['Before genesis', ((S.genesis - t0) / 3600).toFixed(1) + ' h']);
    showTip(evt, fmtTime(t0) + ' to ' + fmtTime(t0 + st.bin * S.dt), rows);
  }

  // ---------------------------------------------------------------- control
  function segmented(id, options, get, set) {
    var box = clear($(id));
    options.forEach(function (o) {
      var b = el('button', { type: 'button', role: 'radio', 'aria-checked': String(get() === o[0]) }, o[1]);
      b.addEventListener('click', function () { set(o[0]); Array.prototype.forEach.call(box.children, function (c, i) { c.setAttribute('aria-checked', String(options[i][0] === get())); }); });
      box.appendChild(b);
    });
  }
  function cursorText() {
    var S = st.S, t = S.t0 + st.cursor * S.dt, txt = fmtTime(t, true);
    if (S.genesis) { var h = (S.genesis - t) / 3600; txt += h > 0.01 ? '  (' + h.toFixed(1) + ' h before genesis)' : '  (genesis)'; }
    $('cursor-out').textContent = txt;
    $('cursor').value = st.cursor;
  }
  function setCursorStep(k) {
    if (!st.S) return;
    st.cursor = Math.max(1, Math.min(st.S.ns, k));
    st.hover = -1;
    cursorText(); renderTiles(); drawMap(); drawDist(); renderTSCursor(); moveTrackMarker();
  }
  function renderAll() {
    if (!st.S) return;
    readColors();
    cursorText(); renderTiles(); renderMapLegend(); drawMap(); drawDist(); renderTS(); moveTrackMarker();
  }
  var playTimer = null;
  function stopPlay() { st.playing = false; if (playTimer) clearInterval(playTimer); playTimer = null; $('play').textContent = 'Play'; $('play').setAttribute('aria-label', 'Play'); }
  function togglePlay() {
    if (st.playing) return stopPlay();
    if (!st.S) return;
    st.playing = true; $('play').textContent = 'Pause'; $('play').setAttribute('aria-label', 'Pause');
    if (st.cursor >= st.S.ns) st.cursor = Math.min(st.S.ns, Math.max(1, windowSteps(st.S)));
    playTimer = setInterval(function () {
      if (!st.S || st.cursor >= st.S.ns) return stopPlay();
      setCursorStep(st.cursor + 2);
    }, 60);
  }

  function showEmpty(title, body, link) {
    var box = clear($('empty'));
    box.hidden = false;
    box.appendChild(el('h2', null, title));
    box.appendChild(el('p', null, body));
    if (link) { var p = el('p'); p.appendChild(el('a', { href: link[0] }, link[1])); box.appendChild(p); }
  }
  function stamp() {
    var u = st.index && st.index.updated;
    $('stamp').textContent = u ? 'updated ' + fmtTime(Date.parse(u.replace('Z', ':00Z')) / 1000, true) : '';
  }
  function loadIndex(first) {
    return getJSON('index.json').then(function (index) {
      var changed = !st.index || st.index.updated !== index.updated;
      st.index = index;
      stamp();
      var years = index.storms.map(function (e) { return e.year; }).filter(function (v, i, a) { return a.indexOf(v) === i; }).sort().reverse();
      var sel = $('f-year'), keep = sel.value;
      clear(sel).appendChild(el('option', { value: '' }, 'All'));
      years.forEach(function (y) { sel.appendChild(el('option', { value: y }, y)); });
      sel.value = keep;
      if (first) {
        var want = decodeURIComponent(location.hash.replace(/^#/, '')), hit = index.storms.filter(function (e) { return e.id === want; })[0];
        var anyLive = index.storms.some(function (e) { return e.status === 'active'; });
        st.mode = hit ? (hit.status === 'active' ? 'realtime' : 'archive') : (anyLive || !index.storms.length ? 'realtime' : 'archive');
        if (hit) st.id = hit.id;
        setMode(st.mode, !!hit);
        if (hit) select(hit.id);
      } else if (changed) {
        renderList();
        if (st.id && index.storms.some(function (e) { return e.id === st.id; })) select(st.id, true);
      }
    });
  }

  function init() {
    readColors();
    $('demo-banner').hidden = !DEMO;
    segmented('seg-win', WINDOWS, function () { return st.win; }, function (v) { st.win = v; renderAll(); });
    segmented('seg-color', Object.keys(VARS).map(function (k) { return [k, VARS[k].short]; }), function () { return st.colorBy; }, function (v) { st.colorBy = v; renderAll(); });
    segmented('seg-bin', BINS, function () { return st.bin; }, function (v) { st.bin = v; renderAll(); });
    $('inc-land').addEventListener('change', function (e) { st.land = e.target.checked; renderAll(); });
    $('cursor').addEventListener('input', function (e) { stopPlay(); setCursorStep(+e.target.value); });
    $('play').addEventListener('click', togglePlay);
    $('tab-realtime').addEventListener('click', function () { setMode('realtime'); });
    $('tab-archive').addEventListener('click', function () { setMode('archive'); });
    ['f-basin', 'f-outcome', 'f-year'].forEach(function (id) { $(id).addEventListener('change', function () { setMode('archive', true); }); });
    ['map', 'dist'].forEach(function (id) {
      $(id).addEventListener('pointermove', function (e) { hoverCanvas(e, id); });
      $(id).addEventListener('pointerleave', function () { hideTip(); if (st.hover >= 0) { st.hover = -1; drawMap(); drawDist(); } });
    });
    var dragging = false, ts = $('ts');
    ts.addEventListener('pointerdown', function (e) { dragging = true; stopPlay(); try { ts.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } tsPointer(e, true); });
    ts.addEventListener('pointermove', function (e) { tsPointer(e, dragging); });
    ts.addEventListener('pointerup', function () { dragging = false; });
    ts.addEventListener('pointerleave', function () { dragging = false; hideTip(); var h = $('ts-hover'); if (h) clear(h); });
    var pending = null, redraw = function () { if (pending) cancelAnimationFrame(pending); pending = requestAnimationFrame(function () { pending = null; renderAll(); }); };
    if (window.ResizeObserver) new ResizeObserver(redraw).observe($('ts-wrap')); else window.addEventListener('resize', redraw);
    if (window.matchMedia) { var mq = window.matchMedia('(prefers-color-scheme: dark)'); if (mq.addEventListener) mq.addEventListener('change', redraw); }
    if (window.MutationObserver) new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);

    loadCoast().then(function () { return loadIndex(true); }).catch(function (err) {
      if (window.console) console.warn('Pre-Genesis Lightning Monitor: ' + (err && err.message || err));
      $('detail').hidden = true;
      showEmpty('No data has been published yet',
        'This page fills in after the first real-time update or archive run publishes its data.',
        DEMO || EMBED ? null : ['?demo=1', 'Open the page with synthetic example data']);
    });
    if (!EMBED) setInterval(function () { loadIndex(false).catch(function () { /* keep the current view */ }); }, REFRESH_MS);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
