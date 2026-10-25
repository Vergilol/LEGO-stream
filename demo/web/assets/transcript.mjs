/**
 * Pure presentation logic for the transcript: no DOM, no network. The DOM side
 * is `app.mjs`.
 */

/** Ten accent colours, distinct on the demo's dark panel. */
export const SPEAKER_COLORS = [
  "#78e6b0", "#ffc878", "#83c9ff", "#dd9cff", "#ff9f9a",
  "#a9e66f", "#74e0dc", "#f1a6d2", "#d8ce70", "#9baeff",
];

/** Millisecond resolution, so keys survive JSON round-tripping. */
const KEY_DECIMALS = 3;

export function finiteNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function optionalFiniteNumber(value, fallback = null) {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/** Colour for a speaker label, a function of the label alone. */
export function speakerColor(speaker) {
  const size = SPEAKER_COLORS.length;
  const text = String(speaker ?? "");
  const digits = Number.parseInt(text.replace(/\D/g, ""), 10);
  if (Number.isFinite(digits)) {
    // Speaker 1 is the first colour, and "S0" must wrap rather than index -1.
    return SPEAKER_COLORS[((digits - 1) % size + size) % size];
  }
  let hash = 0;
  for (const char of text) {
    hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0;
  }
  return SPEAKER_COLORS[Math.abs(hash) % size];
}

/** Stable identity for a transcript row: kind plus the span it occupies. */
export function segmentKey(kind, segment) {
  const start = finiteNumber(segment?.start).toFixed(KEY_DECIMALS);
  const end = finiteNumber(segment?.end).toFixed(KEY_DECIMALS);
  return `${kind}:${start}|${end}`;
}

/**
 * `mm:ss`, or `mm:ss.d` with `decimals`, truncated as a clock reads. The
 * epsilon keeps `0.3 * 10` from flooring to 2.
 */
export function formatClock(seconds, decimals = 0) {
  const scale = 10 ** decimals;
  const ticks = Math.max(0, Math.floor(finiteNumber(seconds) * scale + 1e-6));
  const minutes = Math.floor(ticks / (60 * scale));
  const remainder = (ticks - minutes * 60 * scale) / scale;
  const width = decimals > 0 ? 3 + decimals : 2;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(decimals).padStart(width, "0")}`;
}

/** A row's span, on the same clock as the Time tile. */
export function formatSpan(segment) {
  return `${formatClock(segment?.start, 1)}–${formatClock(segment?.end, 1)}`;
}

/**
 * The provisional tail, newest first, one lane per voice. The segment is
 * rebuilt from its display fields, so a decoder-local label can never render.
 */
export function selectProvisionalItems(provisional = []) {
  const seen = new Map();
  return (Array.isArray(provisional) ? provisional : [])
    .map((raw) => ({
      lane: Math.max(0, Math.floor(finiteNumber(raw?.lane))),
      segment: {
        start: finiteNumber(raw?.start),
        end: finiteNumber(raw?.end),
        text: String(raw?.text ?? ""),
      },
    }))
    .sort((left, right) => (
      right.segment.start - left.segment.start || right.segment.end - left.segment.end
    ))
    .map(({ lane, segment }) => {
      // Two turns may share a lane and a span; suffix the repeats.
      const base = segmentKey(`provisional${lane}`, segment);
      const repeat = seen.get(base) || 0;
      seen.set(base, repeat + 1);
      return {
        kind: "provisional",
        key: repeat ? `${base}#${repeat + 1}` : base,
        lane,
        segment,
      };
    });
}

/**
 * Every published record, latest start first: timeline order, so a turn
 * committed late lands where it was spoken. Keyed by release position, which
 * is never reused, so two records with one span still get a key each.
 */
export function selectPublishedItems(published = []) {
  const rows = Array.isArray(published) ? published : [];
  return rows
    .map((segment, index) => ({ kind: "published", key: `published:${index}`, index, segment }))
    .sort((left, right) => (
      finiteNumber(right.segment?.start) - finiteNumber(left.segment?.start)
      || finiteNumber(right.segment?.end) - finiteNumber(left.segment?.end)
      || right.index - left.index
    ));
}

/** Keys of the rows whose span contains `time`, end exclusive. */
export function activeKeys(items = [], time = null) {
  const keys = new Set();
  const at = optionalFiniteNumber(time);
  if (at === null) return keys;
  for (const item of Array.isArray(items) ? items : []) {
    const start = finiteNumber(item?.segment?.start);
    const end = finiteNumber(item?.segment?.end);
    if (start <= at && at < end) keys.add(item.key);
  }
  return keys;
}

/** Speakers in order of first release, with how many rows each has. */
export function speakerLegend(published = [], pending = []) {
  const order = [];
  const seen = new Map();
  const note = (segment) => {
    const speaker = String(segment?.speaker ?? "").trim();
    if (!speaker) return;
    const entry = seen.get(speaker);
    if (entry) {
      entry.count += 1;
      return;
    }
    order.push(speaker);
    seen.set(speaker, {count: 1});
  };
  for (const segment of Array.isArray(published) ? published : []) note(segment);
  for (const segment of Array.isArray(pending) ? pending : []) note(segment);
  return order.map((speaker) => ({
    speaker,
    color: speakerColor(speaker),
    count: seen.get(speaker).count,
  }));
}

// --- the append-only check ------------------------------------------------

export function createFingerprintState() {
  return { byKey: new Map(), byIndex: [] };
}

function fieldsOf(segment) {
  return {
    start: finiteNumber(segment?.start),
    end: finiteNumber(segment?.end),
    speaker: String(segment?.speaker ?? ""),
    text: String(segment?.text ?? ""),
  };
}

const FIELD_LABELS = ["start", "end", "speaker", "text"];

function firstDifferingField(previous, current) {
  return FIELD_LABELS.find((field) => previous[field] !== current[field]) ?? null;
}

/**
 * Fold the whole published list (release order) into `state` and return every
 * `{key, index, field, previous, current}` that changed. Idempotent; one state
 * per session.
 *
 * Records are tracked by span key and by position: a changed speaker or text
 * keeps its key, a changed span keeps only its position.
 */
export function checkAppendOnly(state, records = []) {
  const violations = [];
  const rows = Array.isArray(records) ? records : [];
  for (let index = 0; index < rows.length; index += 1) {
    const segment = rows[index];
    const key = segmentKey("published", segment);
    const fields = fieldsOf(segment);
    const known = state.byKey.get(key);

    if (known) {
      const field = firstDifferingField(known, fields);
      if (field) {
        violations.push({
          key, index, field, previous: known[field], current: fields[field],
        });
      }
    }
    state.byKey.set(key, fields);

    const prior = state.byIndex[index];
    if (prior && prior.key !== key) {
      // Same slot, new span: invisible to the keyed map above.
      violations.push({
        key,
        index,
        field: "start/end",
        previous: prior.key,
        current: key,
      });
    }
    state.byIndex[index] = {key, fields};
  }
  return violations;
}

/** The `done` payload must equal what was streamed, record for record. */
export function verifyFinalMatchesStream(finalSegments, streamedSegments) {
  const final = Array.isArray(finalSegments) ? finalSegments : [];
  const streamed = Array.isArray(streamedSegments) ? streamedSegments : [];
  if (final.length !== streamed.length) {
    return {
      ok: false,
      message: `server closed with ${final.length} records but ${streamed.length} were streamed`,
    };
  }
  for (let index = 0; index < final.length; index += 1) {
    const field = firstDifferingField(fieldsOf(streamed[index]), fieldsOf(final[index]));
    if (field) {
      return {
        ok: false,
        message: `record ${index} (${segmentKey("", final[index])}) differs in ${field} from the streamed copy`,
      };
    }
  }
  return {ok: true, message: ""};
}

/**
 * Compare `hypothesis.json` (timeline order) with the stream (release order),
 * as multisets. `null`, an unreadable artifact, is "not checked", never a pass.
 */
export function verifyCommitted(committedSegments, streamedSegments) {
  if (!Array.isArray(committedSegments)) {
    return {status: "unchecked", message: "committed artifact not checked: the server could not read it"};
  }
  const streamed = Array.isArray(streamedSegments) ? streamedSegments : [];
  if (committedSegments.length !== streamed.length) {
    return {
      status: "mismatch",
      message: `the committed artifact holds ${committedSegments.length} records but ${streamed.length} were streamed`,
    };
  }
  // Millisecond spans, so JSON float noise still matches.
  const spanOf = (segment) => segmentKey("", segment);
  const remaining = new Map();
  for (const segment of streamed) {
    const span = spanOf(segment);
    if (!remaining.has(span)) remaining.set(span, []);
    remaining.get(span).push(fieldsOf(segment));
  }
  for (const segment of committedSegments) {
    const span = spanOf(segment);
    const candidates = remaining.get(span) || [];
    const fields = fieldsOf(segment);
    const match = candidates.findIndex((other) => (
      other.speaker === fields.speaker && other.text === fields.text
    ));
    if (match >= 0) {
      candidates.splice(match, 1);
      continue;
    }
    if (candidates.length) {
      const field = fields.speaker !== candidates[0].speaker ? "speaker" : "text";
      return {
        status: "mismatch",
        message: `committed record ${formatSpan(segment)} differs in ${field} from the streamed copy`,
      };
    }
    return {
      status: "mismatch",
      message: `committed record ${formatSpan(segment)} was never streamed`,
    };
  }
  return {status: "ok", message: "matches the committed artifact"};
}

// --- export ---------------------------------------------------------------

function srtTimestamp(seconds) {
  const totalMs = Math.max(0, Math.round(finiteNumber(seconds) * 1000));
  const milliseconds = totalMs % 1000;
  const totalSeconds = Math.floor(totalMs / 1000);
  const secs = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  const stamp = [hours, minutes, secs]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
  return `${stamp},${String(milliseconds).padStart(3, "0")}`;
}

export function buildSrt(segments = []) {
  const ordered = (Array.isArray(segments) ? segments : [])
    .slice()
    .sort((left, right) => (
      finiteNumber(left?.start) - finiteNumber(right?.start)
      || finiteNumber(left?.end) - finiteNumber(right?.end)
    ));
  const blocks = ordered.map((segment, index) => [
    String(index + 1),
    `${srtTimestamp(segment?.start)} --> ${srtTimestamp(segment?.end)}`,
    `[${String(segment?.speaker ?? "")}] ${String(segment?.text ?? "")}`,
  ].join("\n"));
  return blocks.length ? `${blocks.join("\n\n")}\n` : "";
}

export function buildJsonExport(segments = [], {audioName = "audio", status = "idle"} = {}) {
  const rows = (Array.isArray(segments) ? segments : []).map((segment) => {
    const fields = fieldsOf(segment);
    return {
      start: fields.start,
      end: fields.end,
      speaker: fields.speaker,
      text: fields.text,
    };
  });
  const speakers = [...new Set(rows.map((row) => row.speaker))].sort();
  return `${JSON.stringify({
    schema_version: "lego-stream-demo-export.v1",
    audio_name: String(audioName),
    status: String(status),
    duration_sec: rows.reduce((high, row) => Math.max(high, row.end), 0),
    speaker_count: speakers.length,
    speakers,
    segments: rows,
  }, null, 2)}\n`;
}

/** A mono PCM16 WAV of the streamed `frames`, so playback is what the server heard. */
export function buildWav(frames = [], sampleRate = 16000) {
  const parts = Array.isArray(frames) ? frames : [];
  const dataBytes = parts.reduce((total, frame) => total + frame.byteLength, 0);
  const wav = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(wav);
  const ascii = (offset, word) => {
    for (let index = 0; index < word.length; index += 1) {
      view.setUint8(offset + index, word.charCodeAt(index));
    }
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // linear PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, dataBytes, true);
  const bytes = new Uint8Array(wav);
  let offset = 44;
  for (const frame of parts) {
    bytes.set(new Uint8Array(frame), offset);
    offset += frame.byteLength;
  }
  return wav;
}

export function downloadName(audioName, extension) {
  const stem = String(audioName || "transcript")
    .replace(/\.[^./\\]+$/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .slice(0, 80) || "transcript";
  return `${stem}.${extension}`;
}

export function buildLiveWebSocketUrl(locationLike = globalThis.location) {
  if (!locationLike || !locationLike.host) {
    throw new TypeError("a page location with host is required");
  }
  const transport = locationLike.protocol === "https:" ? "wss:" : "ws:";
  return `${transport}//${locationLike.host}/api/live`;
}

/**
 * Reconcile a container's children against `items` by key. Nodes are updated
 * in place and moved only when their position changed, so scroll position and
 * text selection survive a tick.
 */
export function reconcileKeyedChildren(
  container,
  items,
  {keyOf, createNode, updateNode = () => {}} = {},
) {
  if (
    !container
    || !container.children
    || typeof container.insertBefore !== "function"
    || typeof keyOf !== "function"
    || typeof createNode !== "function"
  ) {
    throw new TypeError("keyed reconciliation requires a DOM-like container and callbacks");
  }
  const existingByKey = new Map();
  for (const node of Array.from(container.children)) {
    const key = String(node.dataset?.transcriptKey || "");
    if (key) existingByKey.set(key, node);
  }

  const desiredNodes = [];
  const insertedKeys = [];
  const seenKeys = new Set();
  for (const item of items) {
    const key = String(keyOf(item) || "");
    if (!key) throw new TypeError("keyed reconciliation requires a non-empty key");
    if (seenKeys.has(key)) throw new Error(`duplicate reconciliation key: ${key}`);
    seenKeys.add(key);

    let node = existingByKey.get(key);
    if (node) {
      existingByKey.delete(key);
      updateNode(node, item);
    } else {
      node = createNode(item);
      if (!node || !node.dataset) {
        throw new TypeError("keyed reconciliation createNode must return a DOM-like node");
      }
      node.dataset.transcriptKey = key;
      insertedKeys.push(key);
    }
    desiredNodes.push(node);
  }

  for (let index = 0; index < desiredNodes.length; index += 1) {
    const node = desiredNodes[index];
    const current = container.children[index] || null;
    if (current !== node) container.insertBefore(node, current);
  }

  const removedKeys = [...existingByKey.keys()];
  for (const node of existingByKey.values()) node.remove();
  return {insertedKeys, removedKeys};
}
