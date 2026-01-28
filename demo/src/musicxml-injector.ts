/**
 * Inject fingering annotations back into the MusicXML DOM.
 */

import type { ParsedNote } from "./musicxml-parser";

/**
 * For each note with an assigned finger, inject a <fingering> element
 * into the MusicXML <note> element.
 *
 * Structure injected:
 *   <notations>
 *     <technical>
 *       <fingering>N</fingering>
 *     </technical>
 *   </notations>
 */
export function injectFingerings(
  doc: Document,
  notes: ParsedNote[],
  fingers: number[]
): void {
  for (let i = 0; i < notes.length; i++) {
    const finger = fingers[i];
    if (finger < 1 || finger > 5) continue;

    const el = notes[i].element;

    // Remove ALL existing <fingering> elements from this note
    for (const old of Array.from(el.querySelectorAll("fingering"))) {
      old.parentElement?.removeChild(old);
    }

    // Find or create <notations>
    let notations = el.querySelector("notations");
    if (!notations) {
      notations = doc.createElement("notations");
      el.appendChild(notations);
    }

    // Find or create <technical>
    let technical = notations.querySelector("technical");
    if (!technical) {
      technical = doc.createElement("technical");
      notations.appendChild(technical);
    }

    // Create <fingering> — omit placement to let OSMD position it
    const fingering = doc.createElement("fingering");
    fingering.textContent = String(finger);
    technical.appendChild(fingering);
  }
}

/**
 * Serialize the MusicXML document back to a string.
 */
export function serializeXML(doc: Document): string {
  const serializer = new XMLSerializer();
  let xml = serializer.serializeToString(doc);

  // Ensure XML declaration is present
  if (!xml.startsWith("<?xml")) {
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml;
  }

  return xml;
}
