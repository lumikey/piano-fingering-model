/**
 * Integration tests that run the actual ONNX models.
 * Requires onnxruntime-node and the exported .onnx files in models/.
 */
require("onnxruntime-node");
const { describe, it, before } = require("node:test");
const assert = require("node:assert/strict");
const { loadModels, predictFingerings } = require("../dist/index.js");

let models;

before(async () => {
  models = await loadModels();
});

describe("model loading", () => {
  it("loads both models", () => {
    assert.ok(models.left, "Left model should be loaded");
    assert.ok(models.right, "Right model should be loaded");
  });

  it("caches models on repeated calls", async () => {
    const models2 = await loadModels();
    assert.strictEqual(models.left, models2.left);
    assert.strictEqual(models.right, models2.right);
  });
});

describe("right hand predictions", () => {
  it("predicts valid fingers for a C major scale", async () => {
    // C4 D4 E4 F4 G4 A4 B4 C5
    const midiNotes = [60, 62, 64, 65, 67, 69, 71, 72];
    const notes = midiNotes.map((note, i) => ({
      left: false,
      note,
      time: i * 300,
      duration: 280,
    }));

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 8);
    for (const note of result) {
      assert.ok(note.finger >= 1 && note.finger <= 5,
        `Finger ${note.finger} out of range for MIDI ${note.note}`);
    }
  });

  it("uses more than one finger across a scale", async () => {
    const midiNotes = [60, 62, 64, 65, 67, 69, 71, 72];
    const notes = midiNotes.map((note, i) => ({
      left: false,
      note,
      time: i * 300,
      duration: 280,
    }));

    const result = await predictFingerings(notes, models);
    const uniqueFingers = new Set(result.map((n) => n.finger));

    assert.ok(uniqueFingers.size >= 3,
      `Expected at least 3 different fingers, got ${uniqueFingers.size}: [${[...uniqueFingers]}]`);
  });

  it("assigns different fingers to chord notes", async () => {
    // C major triad, right hand — simultaneous notes should get different fingers
    const notes = [
      { left: false, note: 60, time: 0, duration: 500 }, // C
      { left: false, note: 64, time: 0, duration: 500 }, // E
      { left: false, note: 67, time: 0, duration: 500 }, // G
    ];

    const result = await predictFingerings(notes, models);
    const fingers = result.map((n) => n.finger);
    const uniqueFingers = new Set(fingers);

    assert.equal(uniqueFingers.size, 3,
      `Chord notes should have distinct fingers, got [${fingers}]`);
  });
});

describe("left hand predictions", () => {
  it("predicts valid fingers for a bass line", async () => {
    // C3 G2 A2 F2 — simple bass pattern
    const notes = [
      { left: true, note: 48, time: 0, duration: 400 },
      { left: true, note: 43, time: 500, duration: 400 },
      { left: true, note: 45, time: 1000, duration: 400 },
      { left: true, note: 41, time: 1500, duration: 400 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 4);
    for (const note of result) {
      assert.ok(note.finger >= 1 && note.finger <= 5,
        `Finger ${note.finger} out of range for MIDI ${note.note}`);
    }
  });
});

describe("fixed fingering constraints", () => {
  it("preserves user-specified fingers", async () => {
    const notes = [
      { left: false, note: 60, time: 0, duration: 300, finger: 1 },
      { left: false, note: 62, time: 300, duration: 300 },
      { left: false, note: 64, time: 600, duration: 300, finger: 3 },
      { left: false, note: 65, time: 900, duration: 300 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result[0].finger, 1, "First note should keep fixed finger 1");
    assert.equal(result[2].finger, 3, "Third note should keep fixed finger 3");

    // Predicted notes should still be valid
    assert.ok(result[1].finger >= 1 && result[1].finger <= 5);
    assert.ok(result[3].finger >= 1 && result[3].finger <= 5);
  });
});

describe("both hands", () => {
  it("handles mixed left and right hand input", async () => {
    const notes = [
      { left: true, note: 48, time: 0, duration: 500 },
      { left: false, note: 60, time: 0, duration: 500 },
      { left: true, note: 43, time: 500, duration: 500 },
      { left: false, note: 64, time: 500, duration: 500 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 4);

    const leftResults = result.filter((n) => n.left);
    const rightResults = result.filter((n) => !n.left);

    assert.equal(leftResults.length, 2);
    assert.equal(rightResults.length, 2);

    for (const note of result) {
      assert.ok(note.finger >= 1 && note.finger <= 5);
    }
  });
});
