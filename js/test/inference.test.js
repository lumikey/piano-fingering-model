const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const {
  predictFingerings,
  predictHandFingerings,
} = require("../dist/index.js");

/**
 * Create a mock InferenceSession that always predicts a given finger index.
 * Optionally records calls for inspection.
 */
function createMockSession(fingerIndex = 2) {
  const calls = [];
  return {
    calls,
    run: async (feeds) => {
      const tokenData = feeds.tokens.data;
      const dims = feeds.tokens.dims;
      calls.push({ data: Array.from(tokenData), dims: Array.from(dims) });

      const logits = new Float32Array(5).fill(0);
      logits[fingerIndex] = 1.0;
      return { logits: { data: logits } };
    },
  };
}

describe("predictFingerings", () => {
  it("returns empty array for empty input", async () => {
    const models = { left: createMockSession(), right: createMockSession() };
    const result = await predictFingerings([], models);
    assert.deepStrictEqual(result, []);
  });

  it("predicts fingerings for right hand notes", async () => {
    const models = {
      left: createMockSession(),
      right: createMockSession(0), // always predict thumb (index 0 → finger 1)
    };

    const notes = [
      { left: false, note: 60, time: 0, duration: 500 },
      { left: false, note: 62, time: 500, duration: 500 },
      { left: false, note: 64, time: 1000, duration: 500 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 3);
    for (const note of result) {
      assert.equal(note.finger, 1); // finger index 0 → finger 1
    }
  });

  it("predicts fingerings for left hand notes", async () => {
    const models = {
      left: createMockSession(4), // always predict pinky (index 4 → finger 5)
      right: createMockSession(),
    };

    const notes = [
      { left: true, note: 48, time: 0, duration: 500 },
      { left: true, note: 50, time: 500, duration: 500 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 2);
    for (const note of result) {
      assert.equal(note.finger, 5);
    }
  });

  it("preserves fixed fingerings", async () => {
    const models = {
      left: createMockSession(),
      right: createMockSession(0), // would predict finger 1
    };

    const notes = [
      { left: false, note: 60, time: 0, duration: 500, finger: 3 },
      { left: false, note: 62, time: 500, duration: 500 },
      { left: false, note: 64, time: 1000, duration: 500, finger: 5 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result[0].finger, 3); // preserved
    assert.equal(result[1].finger, 1); // predicted (index 0 → finger 1)
    assert.equal(result[2].finger, 5); // preserved
  });

  it("handles both hands together", async () => {
    const models = {
      left: createMockSession(2),  // predict finger 3
      right: createMockSession(0), // predict finger 1
    };

    const notes = [
      { left: true, note: 48, time: 0, duration: 500 },
      { left: false, note: 60, time: 0, duration: 500 },
      { left: true, note: 50, time: 500, duration: 500 },
      { left: false, note: 62, time: 500, duration: 500 },
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 4);

    const leftNotes = result.filter((n) => n.left);
    const rightNotes = result.filter((n) => !n.left);

    for (const n of leftNotes) assert.equal(n.finger, 3);
    for (const n of rightNotes) assert.equal(n.finger, 1);
  });

  it("returns results sorted by time then note", async () => {
    const models = {
      left: createMockSession(),
      right: createMockSession(),
    };

    // Input in scrambled order
    const notes = [
      { left: false, note: 72, time: 1000, duration: 500 },
      { left: true, note: 40, time: 0, duration: 500 },
      { left: false, note: 60, time: 0, duration: 500 },
      { left: true, note: 48, time: 1000, duration: 500 },
    ];

    const result = await predictFingerings(notes, models);

    // Should be sorted by (time, note)
    for (let i = 1; i < result.length; i++) {
      const prev = result[i - 1];
      const curr = result[i];
      const order = prev.time < curr.time || (prev.time === curr.time && prev.note <= curr.note);
      assert.ok(order, `Note at index ${i} is out of order`);
    }
  });

  it("handles chord notes at the same time", async () => {
    const models = {
      left: createMockSession(),
      right: createMockSession(1), // predict finger 2
    };

    // C major chord, right hand
    const notes = [
      { left: false, note: 64, time: 0, duration: 500 }, // E
      { left: false, note: 60, time: 0, duration: 500 }, // C
      { left: false, note: 67, time: 0, duration: 500 }, // G
    ];

    const result = await predictFingerings(notes, models);

    assert.equal(result.length, 3);

    // Output should be sorted by pitch within the chord
    assert.equal(result[0].note, 60);
    assert.equal(result[1].note, 64);
    assert.equal(result[2].note, 67);
  });

  it("all predicted fingers are in range 1-5", async () => {
    for (let fingerIdx = 0; fingerIdx < 5; fingerIdx++) {
      const models = {
        left: createMockSession(fingerIdx),
        right: createMockSession(fingerIdx),
      };

      const notes = [
        { left: false, note: 60, time: 0, duration: 500 },
        { left: true, note: 48, time: 0, duration: 500 },
      ];

      const result = await predictFingerings(notes, models);

      for (const note of result) {
        assert.ok(note.finger >= 1 && note.finger <= 5,
          `Finger ${note.finger} out of range for model index ${fingerIdx}`);
      }
    }
  });

  it("does not mutate input notes", async () => {
    const models = {
      left: createMockSession(),
      right: createMockSession(),
    };

    const original = { left: false, note: 60, time: 0, duration: 500 };
    const notes = [{ ...original }];

    await predictFingerings(notes, models);

    assert.deepStrictEqual(notes[0], original);
  });
});

describe("predictHandFingerings", () => {
  it("only processes notes matching the specified hand", async () => {
    const session = createMockSession(0);

    const notes = [
      { left: true, note: 48, time: 0, duration: 500 },
      { left: false, note: 60, time: 0, duration: 500 },
      { left: true, note: 50, time: 500, duration: 500 },
    ];

    const result = await predictHandFingerings(notes, true, session);

    assert.equal(result.length, 2);
    for (const note of result) {
      assert.equal(note.left, true);
    }
  });

  it("returns empty array when no notes match the hand", async () => {
    const session = createMockSession();

    const notes = [
      { left: false, note: 60, time: 0, duration: 500 },
    ];

    const result = await predictHandFingerings(notes, true, session);
    assert.deepStrictEqual(result, []);
  });
});

describe("token building", () => {
  it("passes correct tensor shape to model", async () => {
    const session = createMockSession();

    const notes = [
      { left: false, note: 60, time: 0, duration: 500 },
    ];

    await predictHandFingerings(notes, false, session);

    assert.equal(session.calls.length, 1);
    assert.deepStrictEqual(session.calls[0].dims, [1, 26, 5]);
    assert.equal(session.calls[0].data.length, 130); // 1 * 26 * 5
  });

  it("calls model once per note", async () => {
    const session = createMockSession();

    const notes = [
      { left: false, note: 60, time: 0, duration: 500 },
      { left: false, note: 62, time: 500, duration: 500 },
      { left: false, note: 64, time: 1000, duration: 500 },
    ];

    await predictHandFingerings(notes, false, session);

    assert.equal(session.calls.length, 3);
  });

  it("current note token has pitch class and token_type=0.5", async () => {
    const session = createMockSession();

    // Middle C (MIDI 60)
    await predictHandFingerings(
      [{ left: false, note: 60, time: 0, duration: 500 }],
      false,
      session
    );

    const data = session.calls[0].data;

    // Current token is at index 5 (offset 25 in flat array)
    // Feature 0: pitch class of MIDI 60 = (60-21) % 12 = 3, normalized = 3/11
    const pitchClass = data[25];
    assert.ok(
      Math.abs(pitchClass - 3 / 11) < 1e-6,
      `Expected pitch class ${3 / 11}, got ${pitchClass}`
    );

    // Feature 3: token_type should be 0.5
    assert.ok(
      Math.abs(data[28] - 0.5) < 1e-6,
      `Expected token_type 0.5, got ${data[28]}`
    );
  });
});
