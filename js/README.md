# @lumikey/piano-fingering-model

Predicts optimal piano fingerings for a sequence of MIDI notes using a Transformer model. Separate models for left and right hands. Runs via ONNX Runtime in Node.js or the browser.

## Install

```bash
# Node.js
npm install @lumikey/piano-fingering-model onnxruntime-node

# Browser
npm install @lumikey/piano-fingering-model onnxruntime-web
```

## Quick start (Node.js)

```typescript
import { predictFingerings } from "@lumikey/piano-fingering-model";
import { loadModels } from "@lumikey/piano-fingering-model/node";

const models = await loadModels();

const notes = [
  { left: false, note: 60, time: 0, duration: 500 },    // Middle C, right hand
  { left: false, note: 62, time: 500, duration: 500 },   // D
  { left: false, note: 64, time: 1000, duration: 500 },  // E
];

const result = await predictFingerings(notes, models);
// Each note now has a `finger` property (1=thumb, 2=index, 3=middle, 4=ring, 5=pinky)
```

## Browser usage

The main entry point is browser-safe — no Node.js imports. Load the ONNX models yourself and pass them in:

```typescript
import { predictFingerings } from "@lumikey/piano-fingering-model";
import { InferenceSession } from "onnxruntime-web";

const models = {
  left: await InferenceSession.create("/models/fingering_transformer_left.onnx"),
  right: await InferenceSession.create("/models/fingering_transformer_right.onnx"),
};

const result = await predictFingerings(notes, models);
```

The `.onnx` files are included in the package under `models/` and can be copied to your static assets directory.

## Fixed fingerings

You can provide partial fingerings as constraints. Notes with a `finger` value are treated as fixed, and the model fills in the rest:

```typescript
const notes = [
  { left: false, note: 60, time: 0, duration: 500, finger: 1 },    // Force thumb
  { left: false, note: 62, time: 500, duration: 500 },              // Model predicts
  { left: false, note: 64, time: 1000, duration: 500 },             // Model predicts
  { left: false, note: 65, time: 1500, duration: 500, finger: 4 },  // Force ring finger
];

const result = await predictFingerings(notes, models);
```

Fixed fingerings are also visible to the model as lookahead hints, so the predicted notes will account for where the hand needs to be.

## API

### `predictFingerings(notes, models)`

Predict fingerings for a sequence of notes.

- **`notes`** — Array of `Note` objects (see below).
- **`models`** — `{ left: InferenceSession, right: InferenceSession }`.
- **Returns** — `Promise<Note[]>` with `finger` populated on every note.

### `loadModels()` (Node.js only)

Load the bundled ONNX models from the package directory. Imported from the `/node` subpath to keep the main entry point browser-safe. Sessions are cached — subsequent calls return the same instances.

```typescript
import { loadModels } from "@lumikey/piano-fingering-model/node";

const models = await loadModels();
```

### `Note`

```typescript
interface Note {
  left: boolean;      // true = left hand, false = right hand
  note: number;       // MIDI note number (21-108, A0-C8)
  time: number;       // Start time in milliseconds
  duration: number;   // Duration in milliseconds
  velocity?: number;  // MIDI velocity (0-127)
  finger?: number;    // 1-5 (1=thumb, 5=pinky). Optional on input, always present on output.
}
```

## How it works

The model is a Transformer encoder that predicts one finger (1-5) per note. For each note, it sees:

- **5 previous tokens** — the last note played by each finger (MIDI offset, time since release, black/white key)
- **1 current token** — the note being predicted (pitch class, black/white key)
- **20 lookahead tokens** — the next 20 notes (MIDI offset, time until, black/white key, optional finger hint)

Left and right hands use separate models trained independently.
