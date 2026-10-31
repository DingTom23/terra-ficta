"""A static viewer: pan, zoom, and read the gazetteer.

No build step, no server, no dependencies.  The plates are ordinary PNGs on
disk; this file just gives them somewhere to be looked at closely.
"""

from __future__ import annotations

import html
import json


def viewer_html(plates):
    """plates: list of dicts with image, title, subtitle, paras, plate."""
    data = json.dumps(plates, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas — the plates</title>
<style>
  :root {{
    --paper:#efe7d3; --ink:#2e2620; --soft:#6b5c4d;
    --night:#17140f; --panel:#211c16; --rule:#3a322a;
  }}
  * {{ box-sizing:border-box; }}
  html, body {{ height:100%; }}
  body {{
    margin:0; background:var(--night); color:var(--paper);
    font-family: Georgia, 'Times New Roman', serif;
    display:grid; grid-template-columns: 300px 1fr; overflow:hidden;
  }}

  aside {{
    background:var(--panel); border-right:1px solid var(--rule);
    display:flex; flex-direction:column; overflow:hidden;
  }}
  aside h1 {{
    margin:0; padding:26px 22px 6px; font-weight:400; font-size:20px;
    letter-spacing:.30em; text-transform:uppercase;
  }}
  aside .tag {{ padding:0 22px 18px; color:#8d7f70; font-style:italic; font-size:13px; }}
  .list {{ overflow-y:auto; flex:1; padding-bottom:20px; }}
  .item {{
    padding:14px 22px; cursor:pointer; border-left:3px solid transparent;
    display:flex; gap:12px; align-items:baseline;
  }}
  .item:hover {{ background:#2a241d; }}
  .item.on {{ background:#332b22; border-left-color:#c8a56a; }}
  .item .num {{ color:#8d7f70; font-size:12px; letter-spacing:.16em; min-width:52px; }}
  .item .nm {{ font-size:15px; letter-spacing:.04em; }}

  main {{ position:relative; overflow:hidden; cursor:grab; }}
  main.drag {{ cursor:grabbing; }}
  #stage {{ position:absolute; inset:0; }}
  #plate {{
    position:absolute; left:50%; top:50%; transform-origin:center;
    will-change:transform; box-shadow:0 30px 90px rgba(0,0,0,.65);
    user-select:none; -webkit-user-drag:none; max-width:none;
  }}
  #hud {{
    position:absolute; left:0; right:0; bottom:0; padding:20px 26px 22px;
    background:linear-gradient(transparent, rgba(10,8,6,.92) 55%);
    pointer-events:none;
  }}
  #hud h2 {{
    margin:0 0 4px; font-weight:400; font-size:22px; letter-spacing:.14em;
    text-transform:uppercase;
  }}
  #hud .sub {{ margin:0 0 10px; color:#b3a390; font-style:italic; font-size:14px; }}
  #hud p {{ margin:0; max-width:70ch; line-height:1.6; font-size:14px; color:#ded3bf; }}
  .zoom {{ position:absolute; right:20px; top:18px; display:flex; gap:8px; }}
  .zoom button {{
    width:34px; height:34px; border-radius:50%; border:1px solid var(--rule);
    background:rgba(33,28,22,.85); color:var(--paper); font-size:17px; cursor:pointer;
  }}
  .zoom button:hover {{ background:#3a3128; }}
</style>
</head>
<body>
  <aside>
    <h1>Atlas</h1>
    <div class="tag">countries that were never surveyed</div>
    <div class="list" id="list"></div>
  </aside>

  <main id="main">
    <div id="stage"><img id="plate" alt=""></div>
    <div class="zoom">
      <button id="in" title="zoom in">+</button>
      <button id="out" title="zoom out">&minus;</button>
      <button id="fit" title="fit">&#9633;</button>
    </div>
    <div id="hud">
      <h2 id="t"></h2>
      <p class="sub" id="s"></p>
      <p id="p"></p>
    </div>
  </main>

<script>
const PLATES = {data};
const img = document.getElementById('plate');
const main = document.getElementById('main');
const list = document.getElementById('list');
let cur = 0, scale = 1, tx = 0, ty = 0, base = 1;

function fitScale() {{
  const pw = img.naturalWidth || 1, ph = img.naturalHeight || 1;
  const availW = main.clientWidth * 0.94, availH = main.clientHeight * 0.90;
  return Math.min(availW / pw, availH / ph);
}}

function apply() {{
  img.style.transform = `translate(-50%,-50%) translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
}}

function reset() {{
  base = fitScale();
  scale = base; tx = 0; ty = 0; apply();
}}

function show(i) {{
  cur = (i + PLATES.length) % PLATES.length;
  const d = PLATES[cur];
  img.src = d.image;
  document.getElementById('t').textContent = (d.plate ? d.plate + ' · ' : '') + d.title;
  document.getElementById('s').textContent = d.subtitle;
  document.getElementById('p').textContent = d.paras[0];
  [...list.children].forEach((el, k) => el.classList.toggle('on', k === cur));
  img.onload = reset;
}}

PLATES.forEach((d, i) => {{
  const el = document.createElement('div');
  el.className = 'item';
  el.innerHTML = `<span class="num">${{d.plate || ''}}</span><span class="nm">${{d.title}}</span>`;
  el.onclick = () => show(i);
  list.appendChild(el);
}});

main.addEventListener('wheel', e => {{
  e.preventDefault();
  const k = Math.exp(-e.deltaY * 0.0016);
  const next = Math.min(Math.max(scale * k, base * 0.6), base * 12);
  const r = main.getBoundingClientRect();
  const mx = e.clientX - r.left - r.width / 2;
  const my = e.clientY - r.top - r.height / 2;
  tx = mx - (mx - tx) * (next / scale);
  ty = my - (my - ty) * (next / scale);
  scale = next; apply();
}}, {{ passive: false }});

let down = false, lx = 0, ly = 0;
main.addEventListener('pointerdown', e => {{
  down = true; lx = e.clientX; ly = e.clientY; main.classList.add('drag');
}});
window.addEventListener('pointerup', () => {{ down = false; main.classList.remove('drag'); }});
window.addEventListener('pointermove', e => {{
  if (!down) return;
  tx += e.clientX - lx; ty += e.clientY - ly;
  lx = e.clientX; ly = e.clientY; apply();
}});

document.getElementById('in').onclick = () => {{ scale *= 1.35; apply(); }};
document.getElementById('out').onclick = () => {{ scale /= 1.35; apply(); }};
document.getElementById('fit').onclick = reset;

window.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') show(cur + 1);
  if (e.key === 'ArrowLeft') show(cur - 1);
}});
window.addEventListener('resize', reset);

show(0);
</script>
</body>
</html>
"""
