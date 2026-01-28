/**
 * Parse MusicXML into a flat array of notes with MIDI pitch, timing, and hand assignment.
 * Each note retains a reference to its XML Element for later fingering injection.
 */

export interface ParsedNote {
  /** MIDI note number (21–108) */
  midi: number;
  /** true = left hand (staff 2), false = right hand (staff 1) */
  left: boolean;
  /** Absolute start time in milliseconds */
  time: number;
  /** Duration in milliseconds */
  duration: number;
  /** Existing fingering from the score (1–5), if present */
  finger?: number;
  /** Reference to the <note> XML element for injection */
  element: Element;
}

/** Convert MusicXML pitch element to MIDI number. */
function pitchToMidi(pitchEl: Element): number | null {
  const stepEl = pitchEl.querySelector("step");
  const octaveEl = pitchEl.querySelector("octave");
  if (!stepEl || !octaveEl) return null;

  const step = stepEl.textContent!.trim();
  const octave = parseInt(octaveEl.textContent!.trim(), 10);
  const alterEl = pitchEl.querySelector("alter");
  const alter = alterEl ? parseFloat(alterEl.textContent!.trim()) : 0;

  const stepMap: Record<string, number> = {
    C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11,
  };
  const base = stepMap[step];
  if (base === undefined) return null;

  return 12 * (octave + 1) + base + Math.round(alter);
}

/**
 * Parse a MusicXML document and return an array of notes.
 * Handles multiple parts, staves, chords, backup/forward, and tempo markings.
 */
export function parseMusicXML(doc: Document): ParsedNote[] {
  const notes: ParsedNote[] = [];
  const parts = doc.querySelectorAll("part");

  for (const part of parts) {
    let divisions = 1; // ticks per quarter note
    let tempo = 120; // BPM
    let staves = 1;
    let cursor = 0; // current time in ticks
    let prevNoteDuration = 0; // ticks, for chord handling

    const measures = part.querySelectorAll("measure");

    for (const measure of measures) {
      for (const child of measure.children) {
        const tag = child.tagName;

        if (tag === "attributes") {
          const divEl = child.querySelector("divisions");
          if (divEl) divisions = parseInt(divEl.textContent!.trim(), 10);
          const stavesEl = child.querySelector("staves");
          if (stavesEl) staves = parseInt(stavesEl.textContent!.trim(), 10);
        }

        if (tag === "direction") {
          const soundEl = child.querySelector("sound");
          if (soundEl) {
            const t = soundEl.getAttribute("tempo");
            if (t) tempo = parseFloat(t);
          }
        }

        if (tag === "backup") {
          const durEl = child.querySelector("duration");
          if (durEl) cursor -= parseInt(durEl.textContent!.trim(), 10);
        }

        if (tag === "forward") {
          const durEl = child.querySelector("duration");
          if (durEl) cursor += parseInt(durEl.textContent!.trim(), 10);
        }

        if (tag === "note") {
          // Skip grace notes
          if (child.querySelector("grace")) {
            continue;
          }

          // Is this a chord note? (same onset as previous)
          const isChord = child.querySelector("chord") !== null;
          if (!isChord) {
            cursor += prevNoteDuration;
            prevNoteDuration = 0;
          }

          const durEl = child.querySelector("duration");
          const durationTicks = durEl
            ? parseInt(durEl.textContent!.trim(), 10)
            : 0;

          if (!isChord) {
            prevNoteDuration = durationTicks;
          }

          // Skip rests
          if (child.querySelector("rest")) {
            continue;
          }

          // Skip tied continuations — only keep the tie start
          const ties = child.querySelectorAll("tie");
          let isTieStop = false;
          for (const tie of ties) {
            if (tie.getAttribute("type") === "stop") {
              isTieStop = true;
              break;
            }
          }
          if (isTieStop) continue;

          // Get pitch
          const pitchEl = child.querySelector("pitch");
          if (!pitchEl) continue;
          const midi = pitchToMidi(pitchEl);
          if (midi === null) continue;

          // Determine hand from staff number
          const staffEl = child.querySelector("staff");
          const staff = staffEl
            ? parseInt(staffEl.textContent!.trim(), 10)
            : 1;
          // staff 1 = right hand (treble), staff 2 = left hand (bass)
          // For single-staff parts, default to right hand
          const left = staves > 1 && staff === 2;

          // Convert ticks to milliseconds
          const tickMs = 60000 / (tempo * divisions);
          const timeMs = cursor * tickMs;
          const durationMs = durationTicks * tickMs;

          // Extract existing fingering if present
          const fingeringEl = child.querySelector("technical > fingering");
          let finger: number | undefined;
          if (fingeringEl) {
            const f = parseInt(fingeringEl.textContent!.trim(), 10);
            if (f >= 1 && f <= 5) finger = f;
          }

          notes.push({
            midi,
            left,
            time: timeMs,
            duration: durationMs,
            finger,
            element: child,
          });
        }
      }
    }
  }

  // Sort by time then pitch
  notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
  return notes;
}
