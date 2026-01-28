import { InferenceSession, Tensor } from "onnxruntime-common";

const BLACK_KEY_NOTES = new Set([1, 4, 6, 9, 11]); // A#, C#, D#, F#, G#

function isBlackKey(midi: number): number {
  if (midi < 0) return -1.0;
  const keyIndex = midi - 21;
  return BLACK_KEY_NOTES.has(((keyIndex % 12) + 12) % 12) ? 1.0 : 0.0;
}

function midiToPitchClass(midi: number): number {
  if (midi < 0) return -1.0;
  const pitchClass = ((midi - 21) % 12 + 12) % 12;
  return pitchClass / 11.0;
}

interface LookaheadNote {
  midi: number;
  timeUntil: number;
  finger: number | null; // 0-4, or null if unknown
}

/**
 * Build 26x5 input tokens for a single note prediction.
 * Returns a flat Float32Array of 130 values in row-major order.
 */
function buildTokens(
  currentMidi: number,
  fingerLastMidi: number[],
  fingerLastTime: number[],
  lookaheadNotes: LookaheadNote[]
): Float32Array {
  const tokens = new Float32Array(26 * 5);

  // Default: finger_hint = -1 (unknown) for all tokens
  for (let i = 0; i < 26; i++) {
    tokens[i * 5 + 4] = -1.0;
  }

  const refMidiNorm = (currentMidi - 21) / 87.0;

  // Previous tokens (0-4): one per finger
  for (let f = 0; f < 5; f++) {
    const offset = f * 5;
    const midi = fingerLastMidi[f];

    if (midi >= 0) {
      const midiNorm = (midi - 21) / 87.0;
      tokens[offset + 0] = midiNorm - refMidiNorm;
    } else {
      tokens[offset + 0] = -1.0;
    }

    if (fingerLastTime[f] === Infinity) {
      tokens[offset + 1] = 1.0;
    } else {
      tokens[offset + 1] = Math.max(0.0, Math.min(fingerLastTime[f], 10.0)) / 10.0;
    }

    tokens[offset + 2] = midi >= 0 ? isBlackKey(midi) : -1.0;
    tokens[offset + 3] = 0.0; // token_type = previous
  }

  // Current token (index 5)
  const curOffset = 5 * 5;
  tokens[curOffset + 0] = midiToPitchClass(currentMidi);
  tokens[curOffset + 1] = 0.0;
  tokens[curOffset + 2] = isBlackKey(currentMidi);
  tokens[curOffset + 3] = 0.5; // token_type = current

  // Lookahead tokens (indices 6-25)
  for (let j = 0; j < 20; j++) {
    const tOffset = (6 + j) * 5;

    if (j < lookaheadNotes.length) {
      const ln = lookaheadNotes[j];
      const midi = ln.midi;

      if (midi >= 0) {
        const midiNorm = (midi - 21) / 87.0;
        tokens[tOffset + 0] = midiNorm - refMidiNorm;
      } else {
        tokens[tOffset + 0] = -1.0;
      }

      if (ln.timeUntil < 0) {
        tokens[tOffset + 1] = -1.0;
      } else {
        tokens[tOffset + 1] = Math.min(ln.timeUntil / 1000.0, 10.0) / 10.0;
      }

      tokens[tOffset + 2] = midi >= 0 ? isBlackKey(midi) : -1.0;
      tokens[tOffset + 3] = 1.0; // token_type = lookahead

      if (ln.finger !== null && ln.finger >= 0 && ln.finger <= 4) {
        tokens[tOffset + 4] = ln.finger / 4.0;
      } else {
        tokens[tOffset + 4] = -1.0;
      }
    } else {
      tokens[tOffset + 0] = -1.0;
      tokens[tOffset + 1] = -1.0;
      tokens[tOffset + 2] = -1.0;
      tokens[tOffset + 3] = -1.0;
      tokens[tOffset + 4] = -1.0;
    }
  }

  return tokens;
}

function argmax(arr: ArrayLike<number>): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

export interface Note {
  left: boolean;
  note: number;
  time: number;
  duration: number;
  velocity?: number;
  finger?: number;
}

/**
 * Predict fingerings for all notes of one hand.
 *
 * Notes with finger 1-5 are treated as fixed; notes without finger are predicted.
 * Fixed fingers in lookahead are passed to the model as hints.
 */
export async function predictHandFingerings(
  notes: Note[],
  isLeft: boolean,
  model: InferenceSession
): Promise<Note[]> {
  const handNotes = notes.filter((n) => n.left === isLeft);

  if (handNotes.length === 0) return [];

  // Sort by (time, midi) — chord notes ordered lowest-to-highest
  const sorted = [...handNotes].sort((a, b) => a.time - b.time || a.note - b.note);

  // Per-finger state
  const fingerLastMidi = [-1, -1, -1, -1, -1];
  const fingerLastTime = [Infinity, Infinity, Infinity, Infinity, Infinity];

  const results: Note[] = [];

  for (let i = 0; i < sorted.length; i++) {
    const note = sorted[i];
    const currentTime = note.time;
    const currentMidi = note.note;

    // Build lookahead from next 20 notes
    const lookahead: LookaheadNote[] = [];
    const end = Math.min(i + 21, sorted.length);
    for (let k = i + 1; k < end; k++) {
      const futureNote = sorted[k];
      const timeUntil = futureNote.time - currentTime;

      const futureFinger = futureNote.finger;
      const fingerHint =
        futureFinger !== undefined && futureFinger >= 1 && futureFinger <= 5
          ? futureFinger - 1
          : null;

      lookahead.push({
        midi: futureNote.note,
        timeUntil,
        finger: fingerHint,
      });
    }

    // Build tokens and run model
    const tokenData = buildTokens(currentMidi, fingerLastMidi, fingerLastTime, lookahead);
    const inputTensor = new Tensor("float32", tokenData, [1, 26, 5]);
    const output = await model.run({ tokens: inputTensor });
    const logits = output.logits.data as Float32Array;
    const pred = argmax(logits);

    const result = { ...note };

    // Keep fixed fingering if provided, otherwise use prediction
    const existing = note.finger;
    let finger: number;
    if (existing !== undefined && existing >= 1 && existing <= 5) {
      finger = existing;
    } else {
      finger = pred + 1; // Convert 0-4 to 1-5
      result.finger = finger;
    }

    results.push(result);

    // Update finger state
    const fingerIdx = finger - 1;
    fingerLastMidi[fingerIdx] = currentMidi;
    const durationSec = (note.duration ?? 100) / 1000.0;
    fingerLastTime[fingerIdx] = -durationSec; // negative = still holding

    // Age finger times for next note
    if (i + 1 < sorted.length) {
      const dt = (sorted[i + 1].time - currentTime) / 1000.0;
      for (let f = 0; f < 5; f++) {
        if (fingerLastTime[f] !== Infinity) {
          fingerLastTime[f] += dt;
        }
      }
    }
  }

  return results;
}

export type Models = { left: InferenceSession; right: InferenceSession };

let cachedModels: Models | null = null;

/**
 * Load the bundled ONNX models from the package directory.
 *
 * Works in Node.js. For browser usage, create InferenceSession objects
 * manually and pass them to `predictFingerings()` instead.
 *
 * Sessions are cached after the first call.
 */
export async function loadModels(): Promise<Models> {
  if (cachedModels) return cachedModels;

  const fs = await import("fs/promises");
  const path = await import("path");

  const modelsDir = path.resolve(__dirname, "..", "models");

  const [leftBuf, rightBuf] = await Promise.all([
    fs.readFile(path.join(modelsDir, "fingering_transformer_left.onnx")),
    fs.readFile(path.join(modelsDir, "fingering_transformer_right.onnx")),
  ]);

  const [left, right] = await Promise.all([
    InferenceSession.create(leftBuf.buffer as ArrayBuffer),
    InferenceSession.create(rightBuf.buffer as ArrayBuffer),
  ]);

  cachedModels = { left, right };
  return cachedModels;
}

/**
 * Predict fingerings for a sequence of piano notes.
 *
 * Notes with an existing `finger` (1-5) are treated as fixed constraints.
 * Notes without `finger` will have one predicted by the model.
 *
 * If `models` is omitted, the bundled ONNX models are loaded automatically
 * (Node.js only). For browser usage, pass pre-loaded InferenceSession objects.
 *
 * @param notes - Array of notes to predict fingerings for
 * @param models - Optional pre-loaded ONNX inference sessions for left and right hand models
 * @returns All notes with `finger` populated (1=thumb, 5=pinky)
 */
export async function predictFingerings(
  notes: Note[],
  models?: Models
): Promise<Note[]> {
  const m = models ?? (await loadModels());

  // Run sequentially — onnxruntime-web's WASM backend does not support
  // concurrent session.run() calls, and the performance difference is
  // negligible since each hand already runs note-by-note internally.
  const leftResults = await predictHandFingerings(notes, true, m.left);
  const rightResults = await predictHandFingerings(notes, false, m.right);

  const allResults = [...leftResults, ...rightResults];
  allResults.sort((a, b) => a.time - b.time || a.note - b.note);

  return allResults;
}
