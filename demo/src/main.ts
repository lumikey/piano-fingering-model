import "./styles.css";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";
import JSZip from "jszip";
import { parseMusicXML, type ParsedNote } from "./musicxml-parser";
import { injectFingerings, serializeXML } from "./musicxml-injector";
import { runPrediction } from "./model-runner";

// --- DOM elements ---
const dropZone = document.getElementById("drop-zone")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const statusEl = document.getElementById("status")!;
const sheetContainer = document.getElementById("sheet-container")!;
const osmdContainer = document.getElementById("osmd-container")!;
const actionsEl = document.getElementById("actions")!;
const downloadBtn = document.getElementById("download-btn") as HTMLButtonElement;

// --- State ---
let fingeredXml: string | null = null;
let fileName = "fingered.musicxml";

// --- OSMD ---
const osmd = new OpenSheetMusicDisplay(osmdContainer, {
  autoResize: true,
  backend: "svg",
  drawTitle: true,
});

// --- Status helpers ---
function showStatus(msg: string, isError = false, showSpinner = false) {
  statusEl.classList.remove("hidden", "error");
  statusEl.innerHTML = "";
  if (showSpinner) {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    statusEl.appendChild(spinner);
  }
  statusEl.appendChild(document.createTextNode(msg));
  if (isError) statusEl.classList.add("error");
}

function hideStatus() {
  statusEl.classList.add("hidden");
}

// --- File handling ---

/** Read a File as text. */
function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

/** Read a File as ArrayBuffer. */
function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

/** Extract MusicXML string from an MXL (ZIP) file. */
async function extractMxl(file: File): Promise<string> {
  const buf = await readFileAsArrayBuffer(file);
  const zip = await JSZip.loadAsync(buf);

  // Look for META-INF/container.xml to find the rootfile
  const containerFile = zip.file("META-INF/container.xml");
  if (containerFile) {
    const containerXml = await containerFile.async("string");
    const parser = new DOMParser();
    const containerDoc = parser.parseFromString(containerXml, "text/xml");
    const rootfile = containerDoc.querySelector("rootfile");
    if (rootfile) {
      const fullPath = rootfile.getAttribute("full-path");
      if (fullPath) {
        const entry = zip.file(fullPath);
        if (entry) return entry.async("string");
      }
    }
  }

  // Fallback: find the first .xml or .musicxml file in the zip
  for (const [name, entry] of Object.entries(zip.files)) {
    if (!entry.dir && (name.endsWith(".xml") || name.endsWith(".musicxml"))) {
      // Skip container.xml itself
      if (name.includes("META-INF")) continue;
      return entry.async("string");
    }
  }

  throw new Error("No MusicXML file found inside the MXL archive");
}

/** Get MusicXML string from any supported file type. */
async function getXmlString(file: File): Promise<string> {
  const name = file.name.toLowerCase();
  if (name.endsWith(".mxl")) {
    return extractMxl(file);
  }
  return readFileAsText(file);
}

// --- Core pipeline ---

async function processXmlString(xmlString: string, name: string) {
  fingeredXml = null;
  downloadBtn.disabled = true;
  actionsEl.classList.add("hidden");
  fileName = name;

  // 1. Parse and render immediately (no fingerings yet)
  showStatus("Rendering sheet music...", false, true);
  await osmd.load(xmlString);
  osmd.render();
  sheetContainer.classList.remove("hidden");

  // 2. Parse for model input
  showStatus("Running fingering model...", false, true);
  const parser = new DOMParser();
  const doc = parser.parseFromString(xmlString, "text/xml");

  const parseError = doc.querySelector("parsererror");
  if (parseError) {
    throw new Error("Invalid XML: " + parseError.textContent);
  }

  const parsedNotes: ParsedNote[] = parseMusicXML(doc);
  if (parsedNotes.length === 0) {
    showStatus("No notes found in the file.", true);
    return;
  }

  // 3. Run model
  const fingers = await runPrediction(parsedNotes);

  // 4. Inject fingerings into XML DOM
  showStatus("Injecting fingerings...", false, true);
  injectFingerings(doc, parsedNotes, fingers);
  fingeredXml = serializeXML(doc);

  // 5. Re-render with fingerings
  showStatus("Rendering fingerings...", false, true);
  await osmd.load(fingeredXml);
  osmd.render();

  // 6. Enable download
  actionsEl.classList.remove("hidden");
  downloadBtn.disabled = false;

  showStatus(
    `Done — predicted fingerings for ${parsedNotes.length} notes.`
  );
}

async function processFile(file: File) {
  try {
    showStatus("Reading file...", false, true);
    const xmlString = await getXmlString(file);
    const name = file.name.replace(/\.(mxl|xml|musicxml)$/i, "_fingered.musicxml");
    await processXmlString(xmlString, name);
  } catch (err) {
    console.error(err);
    showStatus(
      err instanceof Error ? err.message : "An unexpected error occurred.",
      true
    );
  }
}

async function loadDefaultFile() {
  try {
    const url = `${import.meta.env.BASE_URL}default.musicxml`;
    const resp = await fetch(url);
    if (!resp.ok) return; // no default file available, silently skip
    const xmlString = await resp.text();
    await processXmlString(xmlString, "arabesque_fingered.musicxml");
  } catch (err) {
    console.error("Failed to load default file:", err);
  }
}

// --- Download ---

function downloadXml() {
  if (!fingeredXml) return;
  const blob = new Blob([fingeredXml], { type: "application/xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// --- Event listeners ---

// Drag and drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    processFile(files[0]);
  }
});

// Click on drop zone opens file picker
dropZone.addEventListener("click", (e) => {
  // Don't trigger if clicking the label/button inside
  if ((e.target as HTMLElement).closest(".file-label")) return;
  fileInput.click();
});

// File input change
fileInput.addEventListener("change", () => {
  const files = fileInput.files;
  if (files && files.length > 0) {
    processFile(files[0]);
    fileInput.value = ""; // allow re-uploading same file
  }
});

// Download button
downloadBtn.addEventListener("click", downloadXml);

// Load default file on startup
loadDefaultFile();
