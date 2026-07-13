import json
from collections import Counter

CENTER_LAT = 45.4772741
CENTER_LON = 9.2285005
RADIUS_M = 1000

with open("alberi_filtered.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

feats = data["features"]

# genus counts (ranked)
counts = Counter()
for ft in feats:
    g = ft["properties"].get("genere") or "Sconosciuto"
    counts[g] += 1
ranked = [g for g, _ in counts.most_common()]

# distinct color per genus using golden-angle HSL, alternating lightness
def color_for(i, n):
    hue = (i * 137.508) % 360
    light = 45 if i % 2 == 0 else 60
    sat = 70 if i % 3 else 85
    return f"hsl({hue:.0f},{sat}%,{light}%)"

color_map = {g: color_for(i, len(ranked)) for i, g in enumerate(ranked)}

# slim point records
points = []
for ft in feats:
    p = ft["properties"]
    lon, lat = ft["geometry"]["coordinates"][0], ft["geometry"]["coordinates"][1]
    g = p.get("genere") or "Sconosciuto"
    points.append({
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "g": g,
        "s": p.get("specie") or "",
        "d": p.get("diam_tronc") or "",
        "c": p.get("diam_chiom") or "",
        "h": p.get("h_m") or "",
        "y": (p.get("data_ini") or "")[:4],
    })

legend = [{"g": g, "n": counts[g], "c": color_map[g]} for g in ranked]

payload = {
    "center": [CENTER_LAT, CENTER_LON],
    "radius": RADIUS_M,
    "points": points,
    "legend": legend,
    "colors": color_map,
}

html = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Alberi di Milano — entro 1 km</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #map{position:absolute;top:0;left:0;right:0;bottom:0}
  .panel{position:absolute;top:12px;right:12px;z-index:1000;background:rgba(255,255,255,.96);
    border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.25);max-height:calc(100% - 24px);
    width:250px;display:flex;flex-direction:column;overflow:hidden}
  .panel h1{font-size:14px;margin:0;padding:12px 14px 6px;color:#1a3d1a}
  .panel .sub{font-size:11px;color:#666;padding:0 14px 8px}
  .search{margin:0 12px 8px;padding:6px 8px;border:1px solid #ccc;border-radius:6px;font-size:12px}
  .legend{overflow-y:auto;padding:0 6px 10px}
  .row{display:flex;align-items:center;gap:8px;padding:3px 8px;cursor:pointer;border-radius:6px;font-size:12px}
  .row:hover{background:#f0f0f0}
  .row.off{opacity:.35}
  .dot{width:12px;height:12px;border-radius:50%;flex:0 0 12px;border:1px solid rgba(0,0,0,.25)}
  .name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .cnt{color:#888;font-variant-numeric:tabular-nums}
  .actions{display:flex;gap:6px;padding:6px 12px}
  .actions button{flex:1;font-size:11px;padding:5px;border:1px solid #ccc;background:#fafafa;border-radius:6px;cursor:pointer}
  .leaflet-popup-content{font-size:12px;line-height:1.5}
  .leaflet-popup-content b{color:#1a3d1a}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h1>Alberi entro 1 km</h1>
  <div class="sub" id="count"></div>
  <input class="search" id="search" placeholder="Filtra genere..."/>
  <div class="actions">
    <button id="all">Tutti</button>
    <button id="none">Nessuno</button>
  </div>
  <div class="legend" id="legend"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __PAYLOAD__;
const map = L.map('map').setView(DATA.center, 15);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20
}).addTo(map);

// center marker + radius
L.circle(DATA.center, {radius: DATA.radius, color:'#c0392b', weight:2, fill:false, dashArray:'6 6'}).addTo(map);
L.circleMarker(DATA.center, {radius:5, color:'#c0392b', fillColor:'#c0392b', fillOpacity:1}).addTo(map)
  .bindPopup('Centro<br>'+DATA.center[0]+', '+DATA.center[1]);

const active = new Set(DATA.legend.map(l=>l.g));
const layers = {}; // genus -> LayerGroup

function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

DATA.legend.forEach(l=>{ layers[l.g] = L.layerGroup(); });
DATA.points.forEach(p=>{
  const col = DATA.colors[p.g];
  const m = L.circleMarker([p.lat,p.lon], {radius:4,color:col,fillColor:col,fillOpacity:.85,weight:1});
  const html = '<b>'+esc(p.g)+(p.s?' '+esc(p.s):'')+'</b><br>'+
    (p.h?'Altezza: '+esc(p.h)+' m<br>':'')+
    (p.d?'Diam. tronco: '+esc(p.d)+' cm<br>':'')+
    (p.c?'Diam. chioma: '+esc(p.c)+' m<br>':'')+
    (p.y?'Impianto: '+esc(p.y):'');
  m.bindPopup(html);
  layers[p.g].addLayer(m);
});
Object.values(layers).forEach(lg=>lg.addTo(map));

// legend UI
const legendEl = document.getElementById('legend');
const rows = {};
DATA.legend.forEach(l=>{
  const row = document.createElement('div');
  row.className='row';
  row.innerHTML = '<span class="dot" style="background:'+l.c+'"></span>'+
    '<span class="name">'+esc(l.g)+'</span><span class="cnt">'+l.n+'</span>';
  row.onclick=()=>toggle(l.g);
  legendEl.appendChild(row);
  rows[l.g]=row;
});
document.getElementById('count').textContent = DATA.points.length+' alberi · '+DATA.legend.length+' generi';

function toggle(g){
  if(active.has(g)){active.delete(g); map.removeLayer(layers[g]); rows[g].classList.add('off');}
  else{active.add(g); layers[g].addTo(map); rows[g].classList.remove('off');}
}
document.getElementById('all').onclick=()=>DATA.legend.forEach(l=>{if(!active.has(l.g))toggle(l.g);});
document.getElementById('none').onclick=()=>DATA.legend.forEach(l=>{if(active.has(l.g))toggle(l.g);});
document.getElementById('search').oninput=e=>{
  const q=e.target.value.toLowerCase();
  DATA.legend.forEach(l=>{rows[l.g].style.display=l.g.toLowerCase().includes(q)?'flex':'none';});
};
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
with open("trees_map.html", "w", encoding="utf-8") as f:
    f.write(html)

import os
print("wrote trees_map.html", round(os.path.getsize("trees_map.html")/1024), "KB")
