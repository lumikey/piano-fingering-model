/**
 * Load ONNX models and run fingering prediction in the browser.
 */

import * as ort from "onnxruntime-web";
import { predictFingerings } from "@lumikey/piano-fingering-model";
import type { Note, Models } from "@lumikey/piano-fingering-model";
import type { ParsedNote } from "./musicxml-parser";

// Point ONNX Runtime WASM to CDN to avoid Vite bundling issues
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

let cachedModels: Models | null = null;

async function getModels(): Promise<Models> {
  if (cachedModels) return cachedModels;

  const modelsBase = `${import.meta.env.BASE_URL}models/`;

  const [left, right] = await Promise.all([
    ort.InferenceSession.create(
      `${modelsBase}fingering_transformer_left.onnx`
    ),
    ort.InferenceSession.create(
      `${modelsBase}fingering_transformer_right.onnx`
    ),
  ]);

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
  }));

  const results = await predictFingerings(inputNotes, models);

  // Results are sorted by (time, note) — same order as parsedNotes
  return results.map((r) => r.finger!);
}
