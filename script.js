/* ==========================================================================
   Room type predictor — frontend logic
   Talks to the FastAPI service: POST /predict  ->  { predicted_room_type, probability }
   ========================================================================== */

// Where the FastAPI server lives. Override without editing: index.html?api=http://host:port
const API_BASE = (new URLSearchParams(location.search).get("api") || "http://127.0.0.1:8000").replace(/\/$/, "");

// Class order of the trained model (scikit-learn sorts labels alphabetically).
// predict_proba columns follow this exact order.
const CLASSES = ["Entire home/apt", "Private room", "Shared room"];
const ROOMS = {
  "Entire home/apt": { key: "home",    name: "Entire home",  icon: "#i-home" },
  "Private room":    { key: "private", name: "Private room", icon: "#i-private" },
  "Shared room":     { key: "shared",  name: "Shared room",  icon: "#i-shared" },
};

// Every neighbourhood the model was trained on, grouped by borough.
const NEIGHBOURHOODS = {
  "Bronx": [
    "Allerton",
    "Baychester",
    "Belmont",
    "Bronxdale",
    "Castle Hill",
    "City Island",
    "Claremont Village",
    "Clason Point",
    "Co-op City",
    "Concourse",
    "Concourse Village",
    "East Morrisania",
    "Eastchester",
    "Edenwald",
    "Fieldston",
    "Fordham",
    "Highbridge",
    "Hunts Point",
    "Kingsbridge",
    "Longwood",
    "Melrose",
    "Morris Heights",
    "Morris Park",
    "Morrisania",
    "Mott Haven",
    "Mount Eden",
    "Mount Hope",
    "North Riverdale",
    "Norwood",
    "Olinville",
    "Parkchester",
    "Pelham Bay",
    "Pelham Gardens",
    "Port Morris",
    "Riverdale",
    "Schuylerville",
    "Soundview",
    "Spuyten Duyvil",
    "Throgs Neck",
    "Tremont",
    "Unionport",
    "University Heights",
    "Van Nest",
    "Wakefield",
    "West Farms",
    "Westchester Square",
    "Williamsbridge",
    "Woodlawn"
  ],
  "Brooklyn": [
    "Bath Beach",
    "Bay Ridge",
    "Bedford-Stuyvesant",
    "Bensonhurst",
    "Bergen Beach",
    "Boerum Hill",
    "Borough Park",
    "Brighton Beach",
    "Brooklyn Heights",
    "Brownsville",
    "Bushwick",
    "Canarsie",
    "Carroll Gardens",
    "Clinton Hill",
    "Cobble Hill",
    "Columbia St",
    "Coney Island",
    "Crown Heights",
    "Cypress Hills",
    "Downtown Brooklyn",
    "DUMBO",
    "Dyker Heights",
    "East Flatbush",
    "East New York",
    "Flatbush",
    "Flatlands",
    "Fort Greene",
    "Fort Hamilton",
    "Gowanus",
    "Gravesend",
    "Greenpoint",
    "Kensington",
    "Manhattan Beach",
    "Midwood",
    "Mill Basin",
    "Navy Yard",
    "Park Slope",
    "Prospect Heights",
    "Prospect-Lefferts Gardens",
    "Red Hook",
    "Sea Gate",
    "Sheepshead Bay",
    "South Slope",
    "Sunset Park",
    "Vinegar Hill",
    "Williamsburg",
    "Windsor Terrace"
  ],
  "Manhattan": [
    "Battery Park City",
    "Chelsea",
    "Chinatown",
    "Civic Center",
    "East Harlem",
    "East Village",
    "Financial District",
    "Flatiron District",
    "Gramercy",
    "Greenwich Village",
    "Harlem",
    "Hell's Kitchen",
    "Inwood",
    "Kips Bay",
    "Little Italy",
    "Lower East Side",
    "Marble Hill",
    "Midtown",
    "Morningside Heights",
    "Murray Hill",
    "NoHo",
    "Nolita",
    "Roosevelt Island",
    "SoHo",
    "Stuyvesant Town",
    "Theater District",
    "Tribeca",
    "Two Bridges",
    "Upper East Side",
    "Upper West Side",
    "Washington Heights",
    "West Village"
  ],
  "Queens": [
    "Arverne",
    "Astoria",
    "Bay Terrace",
    "Bayside",
    "Bayswater",
    "Belle Harbor",
    "Bellerose",
    "Breezy Point",
    "Briarwood",
    "Cambria Heights",
    "College Point",
    "Corona",
    "Ditmars Steinway",
    "Douglaston",
    "East Elmhurst",
    "Edgemere",
    "Elmhurst",
    "Far Rockaway",
    "Flushing",
    "Forest Hills",
    "Fresh Meadows",
    "Glendale",
    "Hollis",
    "Holliswood",
    "Howard Beach",
    "Jackson Heights",
    "Jamaica",
    "Jamaica Estates",
    "Jamaica Hills",
    "Kew Gardens",
    "Kew Gardens Hills",
    "Laurelton",
    "Little Neck",
    "Long Island City",
    "Maspeth",
    "Middle Village",
    "Neponsit",
    "Ozone Park",
    "Queens Village",
    "Rego Park",
    "Richmond Hill",
    "Ridgewood",
    "Rockaway Beach",
    "Rosedale",
    "South Ozone Park",
    "Springfield Gardens",
    "St. Albans",
    "Sunnyside",
    "Whitestone",
    "Woodhaven",
    "Woodside"
  ],
  "Staten Island": [
    "Arden Heights",
    "Arrochar",
    "Bay Terrace, Staten Island",
    "Bull's Head",
    "Castleton Corners",
    "Clifton",
    "Concord",
    "Dongan Hills",
    "Eltingville",
    "Emerson Hill",
    "Graniteville",
    "Grant City",
    "Great Kills",
    "Grymes Hill",
    "Howland Hook",
    "Huguenot",
    "Lighthouse Hill",
    "Mariners Harbor",
    "Midland Beach",
    "New Brighton",
    "New Dorp",
    "New Dorp Beach",
    "New Springville",
    "Oakwood",
    "Port Richmond",
    "Prince's Bay",
    "Randall Manor",
    "Rosebank",
    "Rossville",
    "Shore Acres",
    "Silver Lake",
    "South Beach",
    "St. George",
    "Stapleton",
    "Todt Hill",
    "Tompkinsville",
    "Tottenville",
    "West Brighton",
    "Westerleigh",
    "Willowbrook"
  ]
};

const BOROUGHS = [
  { name: "Manhattan",     dot: "#fccc0a", center: [40.7831, -73.9712] },
  { name: "Brooklyn",      dot: "#e8530e", center: [40.6501, -73.9496] },
  { name: "Queens",        dot: "#b933ad", center: [40.7282, -73.7949] },
  { name: "Bronx",         dot: "#00833a", center: [40.8448, -73.8648] },
  { name: "Staten Island", dot: "#0039a6", center: [40.5795, -74.1502] },
];

const NYC_BOUNDS = { latMin: 40.49, latMax: 40.92, lngMin: -74.27, lngMax: -73.68 };

const PRESETS = {
  midtown: { borough: "Manhattan", neighbourhood: "Midtown", latitude: 40.7549, longitude: -73.984,
             price: 250, minimum_nights: 3, number_of_reviews: 40, reviews_per_month: 1.2,
             calculated_host_listings_count: 1, availability_365: 200 },
  bedstuy: { borough: "Brooklyn", neighbourhood: "Bedford-Stuyvesant", latitude: 40.6782, longitude: -73.9442,
             price: 60, minimum_nights: 2, number_of_reviews: 15, reviews_per_month: 0.8,
             calculated_host_listings_count: 1, availability_365: 300 },
  jamaica: { borough: "Queens", neighbourhood: "Jamaica", latitude: 40.7282, longitude: -73.7949,
             price: 35, minimum_nights: 1, number_of_reviews: 5, reviews_per_month: 0.3,
             calculated_host_listings_count: 3, availability_365: 365 },
};

// Same rules as the Pydantic model in main.py
const RULES = {
  latitude:  { label: "Latitude",  min: -90,  max: 90 },
  longitude: { label: "Longitude", min: -180, max: 180 },
  price:     { label: "Price", exclusiveMin: 0 },
  minimum_nights: { label: "Minimum nights", min: 1, max: 365, int: true },
  number_of_reviews: { label: "Total reviews", min: 0, int: true },
  reviews_per_month: { label: "Reviews per month", min: 0 },
  calculated_host_listings_count: { label: "Listings by this host", min: 0, int: true },
  availability_365: { label: "Days available", min: 0, max: 365, int: true },
};

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);

const form = $("form");
const boroughBox = $("boroughs");
const hoodSelect = $("neighbourhood");
const availability = $("availability_365");
const availabilityOut = $("availability_out");
const result = $("result");
const done = $("done");
const submitBtn = $("submit");

let map = null;
let marker = null;

/* ---------------- Borough + neighbourhood controls ---------------- */

function buildBoroughs() {
  boroughBox.innerHTML = BOROUGHS.map((b, i) => `
    <label class="borough" style="--dot:${b.dot}">
      <input type="radio" name="neighbourhood_group" value="${b.name}" ${i === 0 ? "checked" : ""} />
      <span>${b.name}</span>
    </label>`).join("");
}

function currentBorough() {
  return form.querySelector('input[name="neighbourhood_group"]:checked').value;
}

function fillNeighbourhoods(borough, selected) {
  const list = NEIGHBOURHOODS[borough] || [];
  hoodSelect.innerHTML = list.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join("");
  if (selected && list.includes(selected)) hoodSelect.value = selected;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------- Map ---------------- */

function initMap() {
  const el = $("map");
  if (typeof L === "undefined") {
    el.classList.add("is-missing");
    el.textContent = "The map couldn't load. Type the latitude and longitude below instead.";
    return;
  }
  const start = [parseFloat($("latitude").value) || 40.7549, parseFloat($("longitude").value) || -73.984];
  map = L.map(el, { zoomControl: true, scrollWheelZoom: false }).setView(start, 12);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
  }).addTo(map);

  const icon = L.divIcon({ className: "pin-wrap", html: '<div class="pin"></div>', iconSize: [22, 22], iconAnchor: [11, 11] });
  marker = L.marker(start, { icon, draggable: true, keyboard: false }).addTo(map);

  map.on("click", (e) => setCoords(e.latlng.lat, e.latlng.lng, { fromMap: true }));
  marker.on("dragend", () => {
    const p = marker.getLatLng();
    setCoords(p.lat, p.lng, { fromMap: true });
  });
  map.on("focus", () => (map.scrollWheelZoom.enable()));
  map.on("blur", () => (map.scrollWheelZoom.disable()));
}

function setCoords(lat, lng, { fromMap = false, pan = false } = {}) {
  $("latitude").value = lat.toFixed(5);
  $("longitude").value = lng.toFixed(5);
  clearError("latitude");
  clearError("longitude");
  if (map && marker) {
    marker.setLatLng([lat, lng]);
    if (pan) map.flyTo([lat, lng], Math.max(map.getZoom(), 12), { duration: reduceMotion ? 0 : 0.8 });
    if (fromMap) ping();
  }
  checkBounds(lat, lng);
}

function ping() {
  const wrap = marker && marker.getElement();
  if (!wrap || reduceMotion) return;
  wrap.classList.remove("is-dropped");
  void wrap.offsetWidth;
  wrap.classList.add("is-dropped");
}

function checkBounds(lat, lng) {
  const inside = lat >= NYC_BOUNDS.latMin && lat <= NYC_BOUNDS.latMax && lng >= NYC_BOUNDS.lngMin && lng <= NYC_BOUNDS.lngMax;
  $("map-hint").textContent = inside
    ? "Click the map to move the pin, or type coordinates below."
    : "This spot is outside New York City, so the prediction may be unreliable.";
}

/* ---------------- Form values ---------------- */

function applyValues(v) {
  form.querySelector(`input[name="neighbourhood_group"][value="${v.borough}"]`).checked = true;
  fillNeighbourhoods(v.borough, v.neighbourhood);
  ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]
    .forEach((k) => { $(k).value = v[k]; clearError(k); });
  availability.value = v.availability_365;
  syncSlider();
  setCoords(v.latitude, v.longitude, { pan: true });
}

function syncSlider() {
  const pct = (availability.value / 365) * 100;
  availability.style.setProperty("--pct", pct + "%");
  availabilityOut.textContent = availability.value;
}

/* ---------------- Validation ---------------- */

function showError(id, msg) {
  const input = $(id);
  const p = form.querySelector(`.error[data-for="${id}"]`);
  input.setAttribute("aria-invalid", "true");
  if (p) { p.textContent = msg; input.setAttribute("aria-describedby", `err-${id}`); p.id = `err-${id}`; }
}
function clearError(id) {
  const input = $(id);
  const p = form.querySelector(`.error[data-for="${id}"]`);
  input.removeAttribute("aria-invalid");
  if (p) p.textContent = "";
}

function collect() {
  const errors = {};
  const data = {
    neighbourhood_group: currentBorough(),
    neighbourhood: hoodSelect.value,
  };

  for (const [id, r] of Object.entries(RULES)) {
    const raw = $(id).value.trim();
    if (raw === "") { errors[id] = `Enter ${r.label.toLowerCase()}.`; continue; }
    const n = Number(raw);
    if (!Number.isFinite(n)) { errors[id] = "Enter a number."; continue; }
    if (r.int && !Number.isInteger(n)) { errors[id] = "Use a whole number."; continue; }
    if (r.exclusiveMin !== undefined && n <= r.exclusiveMin) { errors[id] = `Must be greater than ${r.exclusiveMin}.`; continue; }
    if (r.min !== undefined && n < r.min) { errors[id] = `Must be at least ${r.min}.`; continue; }
    if (r.max !== undefined && n > r.max) { errors[id] = `Must be ${r.max} or less.`; continue; }
    data[id] = n;
  }
  return { data, errors };
}

/* ---------------- API ---------------- */

async function checkApi() {
  const status = $("status");
  status.dataset.state = "checking";
  $("status-text").textContent = "Checking API…";
  $("retry").hidden = true;
  try {
    const res = await fetch(API_BASE + "/", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    status.dataset.state = "online";
    $("status-text").textContent = "API connected";
  } catch {
    status.dataset.state = "offline";
    $("status-text").textContent = "API not reachable";
    $("retry").hidden = false;
  }
}

async function predict(payload) {
  let res;
  try {
    res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    const err = new Error(`Can't reach the API at ${API_BASE}. Start it with "uvicorn main:app --reload" and try again.`);
    err.kind = "network";
    throw err;
  }

  if (!res.ok) {
    let msg = `The API answered with an error (${res.status}).`;
    try {
      const body = await res.json();
      if (Array.isArray(body.detail) && body.detail.length) {
        const d = body.detail[0];
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "input";
        msg = `The API rejected "${field}": ${d.msg}.`;
      } else if (typeof body.detail === "string") {
        msg = body.detail;
      }
    } catch { /* keep default message */ }
    throw new Error(msg);
  }
  return res.json();
}

/* ---------------- Result rendering ---------------- */

function setState(state) { result.dataset.state = state; }

function describe(top, second) {
  const [name, p] = top;
  const [name2, p2] = second;
  if (p >= 0.75) return "The model is confident in this prediction.";
  if (p - p2 < 0.15) return `A close call between ${ROOMS[name].name.toLowerCase()} and ${ROOMS[name2].name.toLowerCase()}.`;
  return "The model leans this way, but it isn't certain.";
}

function render(apiResponse) {
  const probs = (apiResponse.probability && apiResponse.probability[0]) || [];
  const ranked = CLASSES.map((c, i) => [c, probs[i] ?? 0]).sort((a, b) => b[1] - a[1]);
  const label = CLASSES.includes(apiResponse.predicted_room_type) ? apiResponse.predicted_room_type : ranked[0][0];
  const room = ROOMS[label];

  // Header of the verdict
  result.dataset.room = room.key;
  done.dataset.room = room.key;
  $("verdict-bullet").dataset.room = room.key;
  $("verdict-icon").innerHTML = `<use href="${room.icon}"/>`;
  $("verdict-name").textContent = room.name;
  $("verdict-note").textContent = describe(ranked[0], ranked[1]);
  result.querySelector(".result__panel").style.setProperty("--room", getComputedStyle(done).getPropertyValue("--room"));

  // Probability bars
  const stops = [...document.querySelectorAll("#line .stop")];
  stops.forEach((li) => {
    const cls = CLASSES.find((c) => ROOMS[c].key === li.dataset.room);
    const value = probs[CLASSES.indexOf(cls)] ?? 0;
    li.classList.toggle("is-top", cls === label);
    li.querySelector(".stop__fill").style.width = "0%";
    li._target = value;
  });

  $("raw").textContent = JSON.stringify(apiResponse, null, 2);

  setState("done");
  done.classList.remove("is-new");
  void done.offsetWidth;
  done.classList.add("is-new");

  requestAnimationFrame(() => requestAnimationFrame(() => {
    stops.forEach((li) => {
      li.querySelector(".stop__fill").style.width = (li._target * 100).toFixed(1) + "%";
      countUp(li.querySelector(".stop__pct"), li._target * 100);
    });
  }));
}

function countUp(el, target) {
  if (reduceMotion) { el.textContent = Math.round(target) + "%"; return; }
  const start = performance.now();
  const dur = 900;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(target * eased) + "%";
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function showFailure(message) {
  $("error-body").textContent = message;
  result.dataset.room = "";
  result.querySelector(".result__panel").style.removeProperty("--room");
  setState("error");
}

/* ---------------- Events ---------------- */

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  Object.keys(RULES).forEach(clearError);

  const { data, errors } = collect();
  const ids = Object.keys(errors);
  if (ids.length) {
    ids.forEach((id) => showError(id, errors[id]));
    $(ids[0]).focus();
    return;
  }

  submitBtn.disabled = true;
  submitBtn.classList.add("is-loading");
  submitBtn.querySelector(".btn__label").textContent = "Predicting…";
  setState("loading");
  result.querySelector(".result__panel").style.removeProperty("--room");
  if (window.innerWidth <= 980) result.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });

  try {
    const [response] = await Promise.all([predict(data), new Promise((r) => setTimeout(r, reduceMotion ? 0 : 450))]);
    render(response);
    checkApi();
  } catch (err) {
    showFailure(err.message);
    if (err.kind === "network") checkApi();
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove("is-loading");
    submitBtn.querySelector(".btn__label").textContent = "Predict room type";
  }
});

boroughBox.addEventListener("change", () => {
  const b = BOROUGHS.find((x) => x.name === currentBorough());
  fillNeighbourhoods(b.name);
  setCoords(b.center[0], b.center[1], { pan: true });
});

availability.addEventListener("input", syncSlider);

["latitude", "longitude"].forEach((id) => {
  $(id).addEventListener("change", () => {
    const lat = parseFloat($("latitude").value);
    const lng = parseFloat($("longitude").value);
    if (Number.isFinite(lat) && Number.isFinite(lng) && map && marker) {
      marker.setLatLng([lat, lng]);
      map.panTo([lat, lng]);
      checkBounds(lat, lng);
    }
  });
});

form.addEventListener("input", (e) => {
  if (e.target.id && RULES[e.target.id]) clearError(e.target.id);
});

document.querySelectorAll("[data-preset]").forEach((btn) =>
  btn.addEventListener("click", () => applyValues(PRESETS[btn.dataset.preset]))
);

$("reset").addEventListener("click", () => {
  applyValues(PRESETS.midtown);
  result.dataset.room = "";
  result.querySelector(".result__panel").style.removeProperty("--room");
  setState("empty");
});

$("retry").addEventListener("click", checkApi);

/* ---------------- Init ---------------- */

buildBoroughs();
fillNeighbourhoods("Manhattan", "Midtown");
initMap();
applyValues(PRESETS.midtown);
checkApi();
