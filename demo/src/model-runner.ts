/**
 * Load ONNX models and run fingering prediction in the browser.
 */

import * as ort from "onnxruntime-web";
import { predictHandFingerings } from "@lumikey/piano-fingering-model";
import type { Note, Models } from "@lumikey/piano-fingering-model";
import type { ParsedNote } from "./musicxml-parser";

// Point ONNX Runtime WASM to CDN to avoid Vite bundling issues
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

let cachedModels: Models | null = null;

async function getModels(): Promise<Models> {
  if (cachedModels) return cachedModels;

  const modelsBase = `${import.meta.env.BASE_URL}models/`;

  // Load sequentially — WASM backend doesn't support concurrent session creation
  const left = await ort.InferenceSession.create(
    `${modelsBase}fingering_transformer_left.onnx`
  );
  const right = await ort.InferenceSession.create(
    `${modelsBase}fingering_transformer_right.onnx`
  );

  cachedModels = {
    left: left as unknown as Models["left"],
    right: right as unknown as Models["right"],
  };
  return cachedModels;
}

/**
 * Convert ParsedNote[] to the Note[] format expected by the model,
 * run prediction, and return finger assignments (1-5) in the same order.
 */
export async function runPrediction(parsedNotes: ParsedNote[]): Promise<number[]> {
  const models = await getModels();

  const inputNotes: Note[] = parsedNotes.map((n) => ({
    left: n.left,
    note: n.midi,
    time: n.time,
    duration: n.duration,
    ...(n.finger !== undefined && { finger: n.finger }),
  }));

  // Run hands sequentially — WASM backend doesn't support concurrent session.run()
  const rightResults = await predictHandFingerings(inputNotes, false, models.right);
  const leftResults = await predictHandFingerings(inputNotes, true, models.left);

  const allResults = [...leftResults, ...rightResults];
  allResults.sort((a, b) => a.time - b.time || a.note - b.note);

  return allResults.map((r) => r.finger!);
}
