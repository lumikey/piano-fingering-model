# Piano Fingering Model

A Transformer model that predicts optimal piano fingerings for MIDI note sequences. Given a series of notes with timing and hand assignment, the model outputs a finger (1-5) for each note.

Separate models are trained for left and right hands. The model supports "fill in the blanks" — you can fix some fingerings and let the model predict the rest, taking your constraints into account.

## Project structure

```
python/              Python training and inference
  model.py           Model architecture (FingeringTransformer)
  train.py           Training script
  inference.py       Batch inference from JSON files
  export_onnx.py     Export PyTorch checkpoints to ONNX format
js/                  TypeScript npm package (@lumikey/piano-fingering-model)
demo/                Browser demo (GitHub Pages)
checkpoints/         Trained model weights (.pt)
data/                Training datasets (.npz)
```

## Model architecture

The model is a Transformer encoder that predicts one finger per note. For each note, the input is a sequence of 26 tokens (5 features each):

| Tokens | Count | Description |
|--------|-------|-------------|
| Previous | 5 | Last note played by each finger: MIDI offset from current note, time since release, black/white key |
| Current | 1 | The note being predicted: pitch class, black/white key |
| Lookahead | 20 | Next 20 notes: MIDI offset, time until, black/white key, optional finger hint |

Default hyperparameters: `d_model=64`, `nhead=4`, `num_layers=3`, `dim_feedforward=128` (~100K parameters).

Output: 5 logits (one per finger), argmax selects the predicted finger.

## Training

Requires Python 3.10+ with PyTorch and NumPy.

```bash
cd python

# Train both hands
python train.py

# Train one hand with custom hyperparameters
python train.py --hand right --d_model 64 --num_layers 3 --epochs 300
```

Training uses:
- AdamW optimizer with weight decay
- Cross-entropy loss with label smoothing (0.1)
- Data augmentation via random masking of finger hints in lookahead tokens
- Learning rate reduction on plateau
- Early stopping (patience=50 epochs)

Trained checkpoints are saved to `checkpoints/`.

### Training data

The `data/` directory contains preprocessed datasets in `.npz` format:

- `fingering_data_{hand}.npz` — Base fingering dataset
- `pig_fingering_data_{hand}.npz` — PIG dataset
- `scale_fingering_data_{hand}.npz` — Scale patterns

All available datasets are concatenated during training.

## Python inference

```bash
cd python
python inference.py notes.json -o output.json
```

Input: JSON array of note objects:

```json
[
  {"left": true, "note": 60, "time": 0, "duration": 500, "velocity": 64},
  {"left": true, "note": 62, "time": 500, "duration": 500, "velocity": 64}
]
```

Notes with a `finger` field (1-5) are treated as fixed constraints. Notes without `finger` are predicted by the model.

## Demo

Try it in the browser: [lumikey.github.io/piano-fingering-model](https://lumikey.github.io/piano-fingering-model/)

Upload a MusicXML file to see predicted fingerings rendered on sheet music.

## JavaScript / TypeScript

The model is available as an npm package for use in Node.js or the browser. The main entry point is browser-safe — Node.js helpers are in a separate `/node` subpath. See [js/README.md](js/README.md) for full documentation.

```bash
npm install @lumikey/piano-fingering-model onnxruntime-node
```

```typescript
import { predictFingerings } from "@lumikey/piano-fingering-model";
import { loadModels } from "@lumikey/piano-fingering-model/node";

const models = await loadModels();
const result = await predictFingerings([
  { left: false, note: 60, time: 0, duration: 500 },
  { left: false, note: 62, time: 500, duration: 500 },
], models);
```

### Exporting models for JS

To update the ONNX models shipped with the npm package after retraining:

```bash
cd python
python export_onnx.py
```

This reads the `.pt` checkpoints and writes `.onnx` files to `js/models/`.
