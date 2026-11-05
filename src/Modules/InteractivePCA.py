'''
Interactive PCA of the test set as a single, self-contained HTML file (PCA_Interactive.html in the run folder).
It needs no additional packages and no internet connection: the data, the image previews and the plotting code
(plain JavaScript on a canvas) are all inside the file. Open it in any web browser.
  - switch between the PCA of the embeddings and of the DINO features
  - zoom with the mouse wheel, pan by dragging, double click to reset
  - show or hide materials and surface states, or show only the misclassified or only the correctly classified points,
    only those with a wrong or only those with the correct surface state, or only those with the correct material
  - colour = surface state, marker = material (see Logging.ClassStyle)
  - hover over a point to see the tile, its name, its true and predicted class and the surface state confidence
To keep the file small, the page shows all misclassified test images and a random sample of the correctly classified
ones, each with an image preview. The PCA itself is computed from all test images, so the positions match PCA_*.png.
Previews need the test images in DataSets/<DataSet>/Images/; without them the page shows all test images without previews.
AddNewMaterialPrototype.py writes the same page as PCA_NewPrototypes.html, with all reference tiles of the new prototypes added.
'''
import base64
import io
import json
import os
import numpy as np

from PIL import Image

from . import Logging as LOG

def Thumbnail(Tile: Image.Image, Size: int = 128) -> str:
    # JPEG thumbnail of a tile as a data URI
    Buffer = io.BytesIO()
    Tile.convert("RGB").resize((Size, Size), Image.LANCZOS).save(Buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(Buffer.getvalue()).decode("ascii")

def Preview(Path: str, Size: int) -> str | None:
    # Thumbnail of a tile file, or None if the image is not available
    if not os.path.isfile(Path):
        return None
    with Image.open(Path) as Tile:
        return Thumbnail(Tile, Size)

def WriteInteractivePCA(Controls, TestHandler, Views: list[dict], NumberOfCorrectImages: int, PreviewSize: int = 128,
                        Extra: dict | None = None, FileName: str = "PCA_Interactive.html", Title: str = ""):
    '''
    Views: one entry per PCA, {"Name", "Coordinates" [images, 2], "Variance" [2], "Prototypes" [prototypes, 2] or None}.
    NumberOfCorrectImages: number of correctly classified test images shown (all misclassified ones are shown).
    Extra: further points shown completely, e.g. the reference tiles of new prototypes (AddNewMaterialPrototype.py):
           {"Members": true class per point, "Names", "Coordinates": one [points, 2] array per view, "Previews": data URI per point,
            "Classifications": prototype index per point, "StateProbabilities"}
    '''
    Data = TestHandler.PreparedData
    LookUp = TestHandler.LookUpTable
    Style  = LOG.ClassStyle(TestHandler.Samples, TestHandler.Classes, [LookUp[Index] for Index in range(len(LookUp))] + (Extra["Members"] if Extra else []))
    WrongStateAll, WrongClassOnlyAll = LOG.MisclassifiedMasks(Data)
    MembersAll = Data["Members"]

    # All misclassified images and a seeded random sample of the correct ones, each with a preview
    Wrong     = np.nonzero(WrongStateAll | WrongClassOnlyAll)[0]
    Correct   = np.nonzero(~(WrongStateAll | WrongClassOnlyAll))[0]
    Generator = np.random.default_rng(Controls.Seed)
    Selected  = sorted(set(Wrong.tolist()) | set(Generator.choice(Correct, size=min(NumberOfCorrectImages, len(Correct)), replace=False).tolist()))
    Previews  = {}
    for Index in Selected:
        Sample, _, Class = MembersAll[Index].partition("_")
        Uri = Preview(f"{Controls.DataSetPath}/Images/{Sample}/{Class}/Testing/{Data['Names'][Index]}", PreviewSize)
        if Uri:
            Previews[Index] = Uri
    Shown = [Index for Index in Selected if Index in Previews] or list(range(len(MembersAll))) # without any images: all points, no previews
    Note  = (f"Shows {len(Shown)} of {len(MembersAll)} test images, each with an image preview: all {sum(Index in Previews for Index in Wrong)} misclassified "
             f"and {sum(Index in Previews for Index in Correct)} randomly chosen correctly classified ones. The PCA is computed from all test images."
             if Previews else f"Shows all {len(MembersAll)} test images without previews: the test images were not found in DataSets/<DataSet>/Images/.")

    # One entry per point: test images first, then the extra points
    State  = lambda Name: Name.split("_")[-1]
    Points = [{"Member": MembersAll[Index], "Name": Data["Names"][Index], "True": Data["Labels"][Index], "Predicted": LookUp[Data["Classifications"][Index]],
               "State": Data["PredictedStates"][Index], "Confidence": round(100 * Data["StateProbabilities"][Index], 2),
               "Error": 2 if WrongStateAll[Index] else 1 if WrongClassOnlyAll[Index] else 0, "Preview": Previews.get(Index)} for Index in Shown]
    Coordinates = [View["Coordinates"][Shown] for View in Views]
    if Extra:
        for Member, Name, Label, Confidence, Uri in zip(Extra["Members"], Extra["Names"], Extra["Classifications"], Extra["StateProbabilities"], Extra["Previews"]):
            Predicted = LookUp[Label]
            Points.append({"Member": Member, "Name": Name, "True": Member, "Predicted": Predicted, "State": State(Predicted), "Confidence": round(100 * Confidence, 2),
                           "Error": 2 if State(Predicted) != State(Member) else 1 if Predicted != Member else 0, "Preview": Uri})
        Coordinates = [np.concatenate([Shown_, Added]) for Shown_, Added in zip(Coordinates, Extra["Coordinates"])]
        Note += f" All {len(Extra['Members'])} reference tiles of the new prototypes are shown as well."
    Groups = list(dict.fromkeys(TestHandler.Samples + [LOG.MaterialOf(Point["Member"]) for Point in Points]))

    Page = {
        "Title"      : Title or f"PCA of the test set - {Controls.RunName}",
        "Groups"     : Groups,
        "GroupMarkers": [Style.MaterialMarker.get(Group, "*") for Group in Groups],
        "Classes"    : TestHandler.Classes,
        "ClassColours": [Style.StateColour[Class] for Class in TestHandler.Classes],
        "Group"      : [Groups.index(LOG.MaterialOf(Point["Member"])) for Point in Points],
        "Class"      : [TestHandler.Classes.index(State(Point["Member"])) for Point in Points],
        "Colour"     : [Style(Point["Member"])[0] for Point in Points],
        "Marker"     : [Style(Point["Member"])[1] for Point in Points],
        **{Key: [Point[Key] for Point in Points] for Key in ["Name", "True", "Predicted", "State", "Confidence", "Error"]},
        "RightMaterial": [int(LOG.MaterialOf(Point["Predicted"]) == LOG.MaterialOf(Point["Member"])) for Point in Points], # nearest prototype has the material of the point
        "Previews"   : {str(Position): Point["Preview"] for Position, Point in enumerate(Points) if Point["Preview"]},
        "PreviewNote": Note,
        "Prototypes" : [{"Name": LookUp[Index], "Colour": Style(LookUp[Index])[0], "Marker": Style(LookUp[Index])[1]} for Index in range(len(LookUp))],
        "Views"      : [{"Name": View["Name"], "Variance": [round(100 * float(Value), 2) for Value in View["Variance"]],
                         "X": np.round(Points_[:, 0], 4).tolist(), "Y": np.round(Points_[:, 1], 4).tolist(),
                         "Prototypes": None if View["Prototypes"] is None else np.round(View["Prototypes"], 4).tolist()} for View, Points_ in zip(Views, Coordinates)],
    }
    Html = TEMPLATE.replace("__DATA__", json.dumps(Page, separators=(",", ":")))
    with open(f"{Controls.OutputPath}/{FileName}", "w", encoding="utf-8") as File:
        File.write(Html)

TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive PCA</title>
<style>
  :root { --text: #1f1f1f; --muted: #555555; --grid: #e3e3e3; --axis: #b0b0b0; --surface: #ffffff; --panel: #f6f6f6; }
  body { margin: 0; padding: 16px; font-family: system-ui, sans-serif; color: var(--text); background: var(--surface); }
  h1 { font-size: 20px; margin: 0 0 8px; font-weight: 600; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: center; margin-bottom: 8px; font-size: 14px; }
  .controls fieldset { border: none; padding: 0; margin: 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .controls legend { float: left; font-weight: 600; margin-right: 6px; padding: 0; }
  label { cursor: pointer; display: inline-flex; align-items: center; gap: 4px; }
  .note { font-size: 13px; color: var(--muted); margin: 4px 0 8px; }
  #plot { position: relative; width: 100%; height: 72vh; min-height: 360px; border: 1px solid var(--grid); }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #tip { position: absolute; pointer-events: none; display: none; background: var(--surface); border: 1px solid var(--axis);
         border-radius: 6px; padding: 8px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); max-width: 260px; }
  #tip img { display: block; width: 128px; height: 128px; margin-bottom: 6px; image-rendering: auto; }
  #tip .name { font-weight: 600; word-break: break-all; }
  #tip .wrong { color: #000000; font-weight: 600; }
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="controls">
  <fieldset id="views"><legend>PCA of</legend></fieldset>
  <fieldset id="groups"><legend>Material</legend></fieldset>
  <fieldset id="states"><legend>Surface state</legend></fieldset>
  <fieldset><legend>Show</legend>
    <label><input type="radio" name="filter" value="all" checked> all</label>
    <label><input type="radio" name="filter" value="wrong"> misclassified only</label>
    <label><input type="radio" name="filter" value="state"> wrong surface state only</label>
    <label><input type="radio" name="filter" value="rightstate"> correct surface state only</label>
    <label><input type="radio" name="filter" value="rightmaterial"> correct material only</label>
    <label><input type="radio" name="filter" value="right"> correctly classified only</label>
  </fieldset>
</div>
<div class="note" id="note"></div>
<div id="plot"><canvas id="canvas"></canvas><div id="tip"></div></div>
<div class="note">Hover over a point to see the tile. Mouse wheel: zoom, drag: pan, double click: reset.
Outlines: black = wrong surface state, grey = wrong class but right surface state. Large symbols with a red outline are the class prototypes.</div>
<script>
const D = __DATA__;
const canvas = document.getElementById("canvas"), tip = document.getElementById("tip"), context = canvas.getContext("2d");
document.getElementById("title").textContent = D.Title;
const N = D.Name.length;
let view = 0, filter = "all", showGroup = D.Groups.map(() => true), showClass = D.Classes.map(() => true);
let zoom = 1, panX = 0, panY = 0, drag = null, hovered = -1, width = 0, height = 0;
const Margin = {left: 64, right: 16, top: 16, bottom: 48};

function addToggle(container, label, swatch, onChange, type = "checkbox", name = "", checked = true) {
  const element = document.createElement("label");
  element.innerHTML = `<input type="${type}" ${name ? `name="${name}"` : ""} ${checked ? "checked" : ""}> ${swatch}${label}`;
  element.querySelector("input").addEventListener("change", event => { onChange(event.target.checked); draw(); });
  document.getElementById(container).appendChild(element);
}
const markerSvg = (marker, colour) => { const p = outline(marker, 7, 7, 5.2);
  return `<svg width="14" height="14">${p ? `<polygon points="${p.map(q => q.join(",")).join(" ")}" fill="${colour}"/>` : `<circle cx="7" cy="7" r="6" fill="${colour}"/>`}</svg>`; };
const swatch = colour => `<svg width="14" height="14"><rect x="1" y="1" width="12" height="12" fill="${colour}"/></svg>`;
D.Views.forEach((v, i) => addToggle("views", v.Name, "", on => { if (on) { view = i; resetView(); } }, "radio", "view", i === 0));
D.Groups.forEach((g, i) => addToggle("groups", g, markerSvg(D.GroupMarkers[i], "#6a6a6a"), on => showGroup[i] = on));
D.Classes.forEach((c, i) => addToggle("states", c, swatch(D.ClassColours[i]), on => showClass[i] = on));
document.querySelectorAll('input[name="filter"]').forEach(input => input.addEventListener("change", e => { filter = e.target.value; draw(); }));
document.getElementById("note").textContent = D.PreviewNote;

function bounds() {
  const v = D.Views[view], xs = v.X.slice(), ys = v.Y.slice();
  if (v.Prototypes) v.Prototypes.forEach(p => { xs.push(p[0]); ys.push(p[1]); });
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = 0.04 * (maxX - minX || 1), padY = 0.04 * (maxY - minY || 1);
  return {minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY};
}
let B = bounds();
function resetView() { zoom = 1; panX = 0; panY = 0; B = bounds(); draw(); }
const plotWidth = () => width - Margin.left - Margin.right, plotHeight = () => height - Margin.top - Margin.bottom;
function toScreen(x, y) {
  const sx = Margin.left + ((x - B.minX) / (B.maxX - B.minX)) * plotWidth(), sy = Margin.top + (1 - (y - B.minY) / (B.maxY - B.minY)) * plotHeight();
  const cx = Margin.left + plotWidth() / 2, cy = Margin.top + plotHeight() / 2;
  return [cx + (sx - cx) * zoom + panX, cy + (sy - cy) * zoom + panY];
}
function fromScreen(px, py) {
  const cx = Margin.left + plotWidth() / 2, cy = Margin.top + plotHeight() / 2;
  const sx = cx + (px - panX - cx) / zoom, sy = cy + (py - panY - cy) / zoom;
  return [B.minX + (sx - Margin.left) / plotWidth() * (B.maxX - B.minX), B.minY + (1 - (sy - Margin.top) / plotHeight()) * (B.maxY - B.minY)];
}
function visible(i) {
  if (!showGroup[D.Group[i]] || !showClass[D.Class[i]]) return false;
  if (filter === "wrong") return D.Error[i] > 0;
  if (filter === "state") return D.Error[i] === 2;
  if (filter === "rightstate") return D.Error[i] !== 2;
  if (filter === "rightmaterial") return D.RightMaterial[i] === 1;
  if (filter === "right") return D.Error[i] === 0;
  return true;
}
function outline(marker, x, y, r) {
  // Corners of the matplotlib markers ^ v < > (triangles), D (diamond), p (pentagon), * (star) and s (square); null for o (circle)
  const polygon = (n, angle, scale, inner = scale) => Array.from({length: n}, (_, k) => { const a = angle + k * 2 * Math.PI / n, s = k % 2 ? inner : scale; return [x + r * s * Math.sin(a), y - r * s * Math.cos(a)]; });
  if (marker === "o") return null;
  if (marker === "^") return polygon(3, 0, 1.15);
  if (marker === "v") return polygon(3, Math.PI, 1.15);
  if (marker === ">") return polygon(3, Math.PI / 2, 1.15);
  if (marker === "<") return polygon(3, -Math.PI / 2, 1.15);
  if (marker === "D") return [[x, y - r * 1.2], [x + r * 0.9, y], [x, y + r * 1.2], [x - r * 0.9, y]];
  if (marker === "p") return polygon(5, 0, 1.1);
  if (marker === "*") return polygon(10, 0, 1.3, 0.55);
  return [[x - r * 0.9, y - r * 0.9], [x + r * 0.9, y - r * 0.9], [x + r * 0.9, y + r * 0.9], [x - r * 0.9, y + r * 0.9]];
}
function shape(marker, x, y, r) {
  context.beginPath();
  const corners = outline(marker, x, y, r);
  if (!corners) context.arc(x, y, r, 0, 2 * Math.PI);
  else { corners.forEach(([px, py], k) => context[k ? "lineTo" : "moveTo"](px, py)); context.closePath(); }
}
function niceStep(range) { const raw = range / 6, power = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / power; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * power; }
function drawAxes() {
  const [x0, y0] = fromScreen(Margin.left, Margin.top + plotHeight()), [x1, y1] = fromScreen(Margin.left + plotWidth(), Margin.top);
  context.font = "12px system-ui, sans-serif"; context.fillStyle = "#555555"; context.strokeStyle = "#e3e3e3"; context.lineWidth = 1;
  const sx = niceStep(x1 - x0), sy = niceStep(y1 - y0);
  context.textAlign = "center"; context.textBaseline = "top";
  for (let x = Math.ceil(x0 / sx) * sx; x <= x1; x += sx) { const [px] = toScreen(x, 0); context.beginPath(); context.moveTo(px, Margin.top); context.lineTo(px, Margin.top + plotHeight()); context.stroke(); context.fillText(+x.toFixed(6), px, Margin.top + plotHeight() + 6); }
  context.textAlign = "right"; context.textBaseline = "middle";
  for (let y = Math.ceil(y0 / sy) * sy; y <= y1; y += sy) { const [, py] = toScreen(0, y); context.beginPath(); context.moveTo(Margin.left, py); context.lineTo(Margin.left + plotWidth(), py); context.stroke(); context.fillText(+y.toFixed(6), Margin.left - 6, py); }
  context.strokeStyle = "#b0b0b0"; context.strokeRect(Margin.left, Margin.top, plotWidth(), plotHeight());
  const v = D.Views[view]; context.fillStyle = "#1f1f1f"; context.font = "14px system-ui, sans-serif";
  context.textAlign = "center"; context.textBaseline = "bottom"; context.fillText(`PC1 (${v.Variance[0]} % of the variance)`, Margin.left + plotWidth() / 2, height - 4);
  context.save(); context.translate(14, Margin.top + plotHeight() / 2); context.rotate(-Math.PI / 2); context.textBaseline = "middle"; context.fillText(`PC2 (${v.Variance[1]} % of the variance)`, 0, 0); context.restore();
}
function draw() {
  context.clearRect(0, 0, width, height); drawAxes();
  context.save(); context.beginPath(); context.rect(Margin.left, Margin.top, plotWidth(), plotHeight()); context.clip();
  const v = D.Views[view];
  for (const level of [0, 1, 2]) {            // correct first, then wrong material, then wrong surface state on top
    for (let i = 0; i < N; i++) {
      if (D.Error[i] !== level || !visible(i)) continue;
      const [x, y] = toScreen(v.X[i], v.Y[i]), r = level === 2 ? 4.5 : level === 1 ? 4.0 : 3.6;
      shape(D.Marker[i], x, y, r);
      context.globalAlpha = level ? 1 : 0.75; context.fillStyle = D.Colour[i]; context.fill(); context.globalAlpha = 1;
      context.lineWidth = level === 2 ? 1.0 : level === 1 ? 0.8 : 0.6; context.strokeStyle = level === 2 ? "#000000" : level === 1 ? "#7a7a7a" : "#ffffff"; context.stroke();
    }
  }
  if (v.Prototypes) v.Prototypes.forEach((p, k) => {
    const proto = D.Prototypes[k], [x, y] = toScreen(p[0], p[1]);
    shape(proto.Marker, x, y, 10);
    context.fillStyle = proto.Colour; context.fill();
    context.lineWidth = 2; context.strokeStyle = "#d00000"; context.stroke();
  });
  if (hovered >= 0) { const [x, y] = toScreen(v.X[hovered], v.Y[hovered]); context.beginPath(); context.arc(x, y, 9, 0, 2 * Math.PI); context.lineWidth = 2; context.strokeStyle = "#1f1f1f"; context.stroke(); }
  context.restore();
}
function resize() {
  const box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  width = box.width; height = box.height; canvas.width = width * ratio; canvas.height = height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
}
function nearest(px, py) {
  const v = D.Views[view]; let best = -1, bestDistance = 64;   // within 8 px
  for (let i = 0; i < N; i++) {
    if (!visible(i)) continue;
    const [x, y] = toScreen(v.X[i], v.Y[i]), d = (x - px) ** 2 + (y - py) ** 2;
    if (d < bestDistance || (d === bestDistance && D.Error[i] > D.Error[best])) { bestDistance = d; best = i; }
  }
  return best;
}
const escape = text => String(text).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
function showTip(i, px, py) {
  const preview = D.Previews[i] ? `<img src="${D.Previews[i]}" alt="">` : "<div class='note'>No preview stored for this image</div>";
  const error = D.Error[i] === 2 ? "<div class='wrong'>Wrong surface state</div>" : D.Error[i] === 1 ? "<div class='wrong'>Wrong class, right surface state</div>" : "";
  tip.innerHTML = `${preview}<div class="name">${escape(D.Name[i])}</div>${error}
    <div>True class: ${escape(D.True[i])}</div><div>Predicted: ${escape(D.Predicted[i])}</div>
    <div>Surface state: ${escape(D.State[i])} (${D.Confidence[i]} %)</div>`;
  tip.style.display = "block";
  const box = document.getElementById("plot").getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = (px + 16 + w > box.width ? px - 16 - w : px + 16) + "px";
  tip.style.top = Math.max(0, Math.min(py - h / 2, box.height - h)) + "px";
}
canvas.addEventListener("mousemove", event => {
  const box = canvas.getBoundingClientRect(), px = event.clientX - box.left, py = event.clientY - box.top;
  if (drag) { panX = drag.panX + px - drag.x; panY = drag.panY + py - drag.y; tip.style.display = "none"; hovered = -1; draw(); return; }
  const i = nearest(px, py);
  if (i !== hovered) { hovered = i; draw(); }
  if (i >= 0) showTip(i, px, py); else tip.style.display = "none";
});
canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; hovered = -1; drag = null; canvas.classList.remove("dragging"); draw(); });
canvas.addEventListener("mousedown", event => { const box = canvas.getBoundingClientRect(); drag = {x: event.clientX - box.left, y: event.clientY - box.top, panX, panY}; canvas.classList.add("dragging"); });
window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
canvas.addEventListener("dblclick", resetView);
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const box = canvas.getBoundingClientRect(), px = event.clientX - box.left, py = event.clientY - box.top;
  const factor = Math.exp(-event.deltaY * 0.0015), newZoom = Math.min(200, Math.max(0.5, zoom * factor)), applied = newZoom / zoom;
  const cx = Margin.left + plotWidth() / 2, cy = Margin.top + plotHeight() / 2;
  panX = (px - cx) - (px - cx - panX) * applied; panY = (py - cy) - (py - cy - panY) * applied; zoom = newZoom; draw();
}, {passive: false});
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
'''
