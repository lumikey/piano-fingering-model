"""
Inference script for fingering prediction.

One prediction per note (not per chord).
Notes within a chord are processed lowest-to-highest.

Supports "fill in the blanks" - provide partial fingerings and the model fills in the rest.
Notes with finger=1-5 are treated as fixed; notes without finger are predicted.

Input: JSON array of notes with format:
  {"left": bool, "note": midi, "time": ms, "finger": 1-5 (optional), "duration": ms}

Output: Same format with predicted finger values.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from model import FingeringTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLACK_KEY_NOTES = [1, 4, 6, 9, 11]  # A#, C#, D#, F#, G# (key_index % 12)


def is_black_key(midi: int) -> float:
    """Return 1.0 if black key, 0.0 if white key, -1.0 if invalid."""
    if midi < 0:
        return -1.0
    key_index = midi - 21
    return 1.0 if (key_index % 12) in BLACK_KEY_NOTES else 0.0


def midi_to_pitch_class(midi: int) -> float:
    """Convert MIDI to pitch class (0-11) normalized to 0-1."""
    if midi < 0:
        return -1.0
    pitch_class = (midi - 21) % 12
    return pitch_class / 11.0


def load_model(hand: str):
    """Load trained fingering transformer."""
    model_path = PROJECT_ROOT / "checkpoints" / f"fingering_transformer_{hand}.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    model = FingeringTransformer(
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        dim_feedforward=checkpoint.get('dim_feedforward', 128)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def build_tokens(current_midi: int, finger_last_midi: list, finger_last_time: list,
                 lookahead_notes: list) -> np.ndarray:
    """Build 26x5 input tokens for a single note prediction."""
    tokens = np.zeros((26, 5), dtype=np.float32)
    tokens[:, 4] = -1.0  # Default: finger_hint unused/unknown

    # Reference: current note
    ref_midi_norm = (current_midi - 21) / 87.0

    # Previous tokens (5): one per finger
    for f in range(5):
        midi = finger_last_midi[f]
        if midi >= 0:
            midi_norm = (midi - 21) / 87.0
            tokens[f, 0] = midi_norm - ref_midi_norm  # offset from current
        else:
            tokens[f, 0] = -1.0  # never used

        # Time since release (negative = still holding)
        if finger_last_time[f] == float('inf'):
            tokens[f, 1] = 1.0  # never used
        else:
            tokens[f, 1] = max(0.0, min(finger_last_time[f], 10.0)) / 10.0

        tokens[f, 2] = is_black_key(int(midi)) if midi >= 0 else -1.0
        tokens[f, 3] = 0.0  # token_type = previous

    # Current token (1)
    tokens[5, 0] = midi_to_pitch_class(current_midi)
    tokens[5, 1] = 0.0  # time = now
    tokens[5, 2] = is_black_key(current_midi)
    tokens[5, 3] = 0.5  # token_type = current

    # Lookahead tokens (20)
    for j in range(20):
        if j < len(lookahead_notes):
            ln = lookahead_notes[j]
            midi = ln['midi']
            if midi >= 0:
                midi_norm = (midi - 21) / 87.0
                tokens[6 + j, 0] = midi_norm - ref_midi_norm
            else:
                tokens[6 + j, 0] = -1.0

            time_until = ln['time_until']
            if time_until < 0:
                tokens[6 + j, 1] = -1.0
            else:
                tokens[6 + j, 1] = min(time_until / 1000.0, 10.0) / 10.0

            tokens[6 + j, 2] = is_black_key(midi) if midi >= 0 else -1.0
            tokens[6 + j, 3] = 1.0  # token_type = lookahead

            # Finger hint: if the note has a fixed finger, include it
            if 'finger' in ln and ln['finger'] is not None and 0 <= ln['finger'] <= 4:
                tokens[6 + j, 4] = ln['finger'] / 4.0
            else:
                tokens[6 + j, 4] = -1.0
        else:
            tokens[6 + j, :] = -1.0

    return tokens


def predict_fingerings(notes: list[dict], hand: str, model, is_left: bool) -> list[dict]:
    """Predict fingerings for all notes of one hand.

    Notes with finger=1-5 are treated as fixed; notes without finger are predicted.
    Fixed fingers in lookahead are passed to the model as hints.
    """
    hand_notes = [n for n in notes if n['left'] == is_left]

    if not hand_notes:
        return []

    # Sort by (time, midi) - this naturally orders chord notes lowest-to-highest
    sorted_notes = sorted(hand_notes, key=lambda n: (n['time'], n['note']))

    # Track per-finger state
    finger_last_midi = [-1.0] * 5
    finger_last_time = [float('inf')] * 5

    results = []

    for i, note in enumerate(sorted_notes):
        current_time = note['time']
        current_midi = note['note']

        # Build lookahead from next 20 notes
        lookahead = []
        for future_note in sorted_notes[i + 1:i + 21]:
            time_until = future_note['time'] - current_time  # 0 for same-chord notes

            # Include finger hint if the note has a fixed finger (1-5 -> 0-4)
            future_finger = future_note.get('finger')
            if future_finger is not None and 1 <= future_finger <= 5:
                finger_hint = future_finger - 1  # Convert 1-5 to 0-4
            else:
                finger_hint = None

            lookahead.append({
                'midi': future_note['note'],
                'time_until': time_until,
                'finger': finger_hint
            })

        # Build tokens and predict
        tokens = build_tokens(current_midi, finger_last_midi, finger_last_time, lookahead)

        with torch.no_grad():
            input_tensor = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
            logits = model(input_tensor)[0]  # (5,)
            pred = logits.argmax().item()

        result_note = note.copy()

        # If note has fixed fingering (1-5), keep it; otherwise use prediction
        existing_finger = note.get('finger')
        if existing_finger is not None and 1 <= existing_finger <= 5:
            finger = existing_finger
        else:
            finger = pred + 1  # Convert 0-4 to 1-5
            result_note['finger'] = finger

        results.append(result_note)

        # Update finger state
        finger_idx = finger - 1
        finger_last_midi[finger_idx] = current_midi
        duration_sec = (note.get('duration') or 100) / 1000.0
        finger_last_time[finger_idx] = -duration_sec  # negative = still holding

        # Age finger times for next note
        if i + 1 < len(sorted_notes):
            dt = (sorted_notes[i + 1]['time'] - current_time) / 1000.0
            for f in range(5):
                if finger_last_time[f] != float('inf'):
                    finger_last_time[f] += dt

    return results


def main():
    parser = argparse.ArgumentParser(description='Predict fingerings for notes')
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output JSON file (default: input_fingered.json)')
    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        notes = json.load(f)

    print(f"Loaded {len(notes)} notes")

    # Load models
    left_model = load_model('left')
    right_model = load_model('right')
    print("Loaded models")

    # Predict for each hand
    left_results = predict_fingerings(notes, 'left', left_model, is_left=True)
    right_results = predict_fingerings(notes, 'right', right_model, is_left=False)

    # Combine and sort by time
    all_results = left_results + right_results
    all_results.sort(key=lambda n: (n['time'], n['note']))

    print(f"Predicted fingerings for {len(all_results)} notes")

    # Save output
    output_path = args.output or args.input.replace('.json', '_fingered.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
