/**
 * The page: DOM, live capture, and playback; the pure logic is in
 * `transcript.mjs`.
 *
 * The microphone, or a file played out loud, goes through one AudioWorklet
 * (16 kHz mono PCM16) and one WebSocket (`WS /api/live`) in real time. The
 * Time tile and every row's span share one clock: the audio captured so far.
 */

import {
  activeKeys,
  buildJsonExport,
  buildLiveWebSocketUrl,
  buildSrt,
  buildWav,
  checkAppendOnly,
  createFingerprintState,
  downloadName,
  finiteNumber,
  formatClock,
  formatSpan,
  optionalFiniteNumber,
  reconcileKeyedChildren,
  selectProvisionalItems,
  selectPublishedItems,
  speakerColor,
  speakerLegend,
  verifyCommitted,
  verifyFinalMatchesStream,
} from "./transcript.mjs";

const $ = (id) => document.getElementById(id);

/** Within this of the top counts as "at the live edge": new rows push down. */
const NEAR_TOP_PX = 8;

const FILE_NOTE = "upload or drop an audio file";

/** 16 kHz mono PCM16, the only format the pipeline accepts. */
const LIVE_SAMPLE_RATE = 16000;
/** 0.1 s frames, as in `pcm-worklet.js`. */
const LIVE_CHUNK_SAMPLES = 1600;
/** A minute of 0.1 s frames waiting to be sent. */
const MAX_PENDING_FRAMES = 600;
const BACKLOG_MESSAGE =
  "the browser fell more than a minute behind the server, so the session was "
  + "stopped rather than silently dropping captured audio";
/** How long the worklet gets to drain its resampler before `eos` goes anyway. */
const FLUSH_TIMEOUT_MS = 1000;
/** `resume()` without a user gesture may wait for one rather than reject. */
const RESUME_TIMEOUT_MS = 1500;

/** Per-session UI state. One of these is live at a time. */
const state = {
  running: false,
  mode: null, // "file" | "mic"
  audioName: "audio",
  status: "idle",
  /** Every published record received, in release order. Never rewritten. */
  published: [],
  /** The latest decoder tail; each tick replaces it wholesale. */
  provisional: [],
  fingerprint: createFingerprintState(),
  metrics: null,
};

/** Why this page cannot stream at all, or `null`. */
let captureBlocker = null;
/** Why the microphone in particular cannot be opened, or `null`. */
let micBlocker = null;

/** The active capture: one per session, see `startCapture`. */
let transport = null;
/** The object URL behind the review player, revoked when it is replaced. */
let playerUrl = null;

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

const dash = "—";

function seconds(value) {
  const number = optionalFiniteNumber(value);
  return number === null ? dash : `${number.toFixed(1)}s`;
}

function ratio(value) {
  const number = optionalFiniteNumber(value);
  return number === null ? dash : number.toFixed(2);
}

function count(value) {
  const number = optionalFiniteNumber(value);
  return number === null ? dash : String(Math.round(number));
}

function say(message, tone = "") {
  const node = $("status");
  node.textContent = message;
  node.className = tone;
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

/**
 * Paint a row, writing only what changed so a text selection survives. A
 * provisional row's speaker cell is an ellipsis: no speaker is decided yet.
 */
function paintRow(node, item, active) {
  const segment = item.segment || {};
  const provisional = item.kind === "provisional";
  const label = provisional ? "" : String(segment.speaker ?? "").trim();
  const [speaker, time, text] = node.children;

  const shown = provisional ? "…" : label || dash;
  if (speaker.textContent !== shown) speaker.textContent = shown;
  const color = label ? speakerColor(label) : "";
  if (speaker.style.color !== color) speaker.style.color = color;

  const span = formatSpan(segment);
  if (time.textContent !== span) time.textContent = span;

  const body = String(segment.text ?? "");
  if (text.textContent !== body) text.textContent = body;

  node.classList.toggle("is-active", active);
  if (active) node.setAttribute("aria-current", "time");
  else node.removeAttribute("aria-current");
}

function createRow(item, active) {
  const row = document.createElement("li");
  row.className = "row";
  const speaker = document.createElement("span");
  speaker.className = "spk";
  const time = document.createElement("span");
  time.className = "t";
  const text = document.createElement("span");
  text.className = "txt";
  row.append(speaker, time, text);
  if (item.kind === "provisional") {
    // The lane is part of the key, so a node never changes lane.
    row.classList.add("provisional");
    row.dataset.lane = String(Math.min(item.lane, 3));
    speaker.title = "no speaker decided yet";
  } else {
    // Motion means "new record", wherever in the timeline it lands.
    row.classList.add("is-new");
    row.addEventListener("animationend", () => row.classList.remove("is-new"), { once: true });
  }
  paintRow(row, item, active);
  return row;
}

function renderRows(list, items, active) {
  reconcileKeyedChildren(list, items, {
    keyOf: (item) => item.key,
    createNode: (item) => createRow(item, active.has(item.key)),
    updateNode: (node, item) => paintRow(node, item, active.has(item.key)),
  });
}

/** The first published row still on screen, to hold the reader's place. */
function anchorRow(scroller) {
  const top = scroller.getBoundingClientRect().top;
  for (const row of $("published").children) {
    if (row.getBoundingClientRect().bottom > top) return row;
  }
  return null;
}

/** Where the review playhead is, or `null` while the player has nothing to say. */
function playheadTime() {
  const player = $("player");
  if ($("playback").hidden || !player.currentSrc) return null;
  if (player.paused && player.currentTime === 0) return null;
  return player.currentTime;
}

function renderLegend(pane, published) {
  const entries = speakerLegend(published);
  pane.hidden = entries.length === 0;
  reconcileKeyedChildren(pane, entries, {
    keyOf: (entry) => entry.speaker,
    createNode: () => {
      const item = document.createElement("li");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      const label = document.createElement("span");
      const total = document.createElement("span");
      total.className = "n";
      item.append(swatch, label, total);
      return item;
    },
    updateNode: (node, entry) => {
      const [swatch, label, total] = node.children;
      swatch.style.background = entry.color;
      label.textContent = entry.speaker;
      label.style.color = entry.color;
      total.textContent = `${entry.count}`;
    },
  });
}

/** Audio captured so far, counted in samples: the clock rows are stamped on. */
function renderTime(value) {
  const number = optionalFiniteNumber(value);
  $("m-time").textContent = number === null ? dash : formatClock(number, 1);
}

function renderMetrics(metrics) {
  if (metrics && typeof metrics === "object") state.metrics = metrics;
  const data = state.metrics || {};
  $("m-rtf").textContent = ratio(data.compute_rtf);
  $("m-delay").textContent = seconds(data.publication_lag_p50_sec);
}

function renderTranscript() {
  const scroller = $("scroller");
  const publishedItems = selectPublishedItems(state.published);
  const provisionalItems = selectProvisionalItems(state.provisional);
  const active = activeKeys([...provisionalItems, ...publishedItems], playheadTime());

  // At the top, new rows push the history down. Scrolled into it, the row in
  // view is held in place instead.
  const pinned = scroller.scrollTop <= NEAR_TOP_PX;
  const anchor = pinned ? null : anchorRow(scroller);
  const before = anchor ? anchor.getBoundingClientRect().top : 0;

  renderRows($("provisional"), provisionalItems, active);
  renderRows($("published"), publishedItems, active);
  $("provisional-caption").hidden = provisionalItems.length === 0;

  if (anchor && anchor.isConnected) {
    scroller.scrollTop += anchor.getBoundingClientRect().top - before;
  }

  // Published rows only: provisional lanes are not speakers.
  renderLegend($("legend"), state.published);
  $("m-speakers").textContent = count(speakerLegend(state.published).length);
  $("count").textContent = String(state.published.length);
  $("empty").hidden = publishedItems.length > 0 || provisionalItems.length > 0;
  $("srt").disabled = state.published.length === 0;
  $("json").disabled = state.published.length === 0;
}

// ---------------------------------------------------------------------------
// the append-only check
// ---------------------------------------------------------------------------

/**
 * Report any published record that changed. Silent while none has; a
 * violation stays on screen for the rest of the session.
 */
function checkContract() {
  const violations = checkAppendOnly(state.fingerprint, state.published);
  if (violations.length === 0) return;
  const [first] = violations;
  const where = first.key || `slot ${first.index}`;
  $("contract").textContent =
    `contract violated: ${where} changed its ${first.field} `
    + `(${JSON.stringify(first.previous)} → ${JSON.stringify(first.current)})`
    + (violations.length > 1 ? `, and ${violations.length - 1} more` : "");
  $("contract").className = "bad";
}

// ---------------------------------------------------------------------------
// one session at a time
// ---------------------------------------------------------------------------

function resetSession(mode, audioName) {
  state.mode = mode;
  state.audioName = audioName || "audio";
  state.published = [];
  state.provisional = [];
  state.fingerprint = createFingerprintState();
  state.metrics = null;
  resetPlayer();
  $("published").replaceChildren();
  $("provisional").replaceChildren();
  $("legend").replaceChildren();
  $("scroller").scrollTop = 0;
  // A microphone has no known end, so its bar sweeps; a file's fills.
  $("progress").classList.toggle("indeterminate", mode === "mic");
  $("fill").style.width = "0";
  renderTime(null);
  renderMetrics(null);
  renderTranscript();
  $("contract").textContent = "";
  $("contract").className = "";
  $("check").textContent = "";
  $("check").className = "";
  $("output").textContent = "";
}

function beginSession(mode, audioName) {
  resetSession(mode, audioName);
  state.running = true;
  state.status = "running";
  // One session at a time, as on the server.
  $("pick").disabled = true;
  $("mic").disabled = true;
  $("stop").hidden = false;
  $("stop").disabled = false;
  $("resume").hidden = true;
  $("note").textContent = "";
}

function endSession() {
  state.running = false;
  $("pick").disabled = Boolean(captureBlocker);
  $("mic").disabled = Boolean(captureBlocker || micBlocker);
  $("stop").hidden = true;
  $("resume").hidden = true;
  $("progress").classList.remove("indeterminate");
  $("level").style.width = "0%";
  $("note").textContent = idleNote();
}

/** What the page can do right now, said before anyone presses a button. */
function idleNote() {
  if (captureBlocker) return captureBlocker;
  if (micBlocker) return `${FILE_NOTE}; ${micBlocker}`;
  return `${FILE_NOTE}, or use the microphone`;
}

function onTick(message) {
  const appended = Array.isArray(message.appended) ? message.appended : [];
  state.published.push(...appended);
  // A snapshot: each tick's tail replaces the last.
  state.provisional = Array.isArray(message.provisional) ? message.provisional : [];

  const expected = optionalFiniteNumber(message.published_count);
  if (expected !== null && expected !== state.published.length) {
    // A lost frame, or a disagreement: either way, say so.
    say(
      `the server reported ${expected} published records but ${state.published.length} arrived`,
      "warn",
    );
  }

  renderMetrics(message.metrics);
  renderTranscript();
  checkContract();
}

/** `sent` is every frame the server accepted, in order: what it heard. */
function onDone(message, sent = []) {
  const final = Array.isArray(message.segments) ? message.segments : [];
  // `segments` restates the stream, so it catches a frame lost in transit;
  // `committed` is the artifact on disk, an independent check. Neither
  // replaces what is on screen.
  const positional = verifyFinalMatchesStream(final, state.published);
  const committed = verifyCommitted(message.committed, state.published);

  // Nothing is provisional once the session is over.
  state.provisional = [];
  state.status = "finished";
  renderMetrics(message.metrics);
  renderTranscript();
  checkContract();

  // Silent when both agree; "not checked" is shown muted, never as a pass.
  const check = $("check");
  if (!positional.ok) {
    check.textContent = positional.message;
    check.className = "bad";
  } else if (committed.status !== "ok") {
    check.textContent = committed.message;
    check.className = committed.status === "mismatch" ? "bad" : "";
  }
  const agreed = positional.ok && committed.status !== "mismatch";
  const validation = message.validation_status;
  if (!agreed) {
    say("finished, but the stream and the committed transcript disagree", "warn");
  } else if (validation !== "passed") {
    say(`finished, but the run's validation is ${JSON.stringify(validation ?? null)}: `
      + "see stream_validation.json next to the transcript", "warn");
  } else {
    say("finished", "ok");
  }

  // The length the server sealed.
  renderTime(message.duration_sec);
  const transcript = String(message.transcript ?? "");
  $("output").textContent = transcript ? `transcript saved to ${transcript}` : "";

  $("fill").style.width = "100%";
  if (sent.length) loadPlayer(buildWav(sent, LIVE_SAMPLE_RATE));
  endSession();
}

/** The session failed. What was published stays: it was never provisional. */
function onError(message) {
  say(String(message || "the session failed"), "err");
  state.provisional = [];
  state.status = "failed";
  endSession();
  renderTranscript();
}

/** Stopped before any audio left the page, so there is nothing to keep. */
function onAbandoned() {
  state.provisional = [];
  state.status = "stopped";
  endSession();
  renderTranscript();
  say("stopped before any audio was streamed; nothing was transcribed", "warn");
}

// ---------------------------------------------------------------------------
// review playback
// ---------------------------------------------------------------------------

function resetPlayer() {
  const player = $("player");
  player.pause();
  player.removeAttribute("src");
  player.load();
  if (playerUrl) URL.revokeObjectURL(playerUrl);
  playerUrl = null;
  $("playback").hidden = true;
  $("player-note").textContent = "";
}

/** The streamed audio, for listening back; the row under the playhead is marked. */
function loadPlayer(wav) {
  playerUrl = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
  $("player").src = playerUrl;
  $("playback").hidden = false;
  $("player-note").textContent = "what the server heard; the row under the playhead is marked";
}

// ---------------------------------------------------------------------------
// capture: a source -> AudioWorklet -> WebSocket
// ---------------------------------------------------------------------------

function captureProblem() {
  if (!globalThis.isSecureContext) {
    return "streaming needs a secure context: use https, or open this page at "
      + "127.0.0.1 or localhost (a plain LAN address is not secure)";
  }
  if (typeof globalThis.AudioWorkletNode !== "function") {
    return "this browser has no AudioWorklet, so it cannot stream audio";
  }
  return null;
}

function microphoneProblem() {
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    return "this browser has no navigator.mediaDevices.getUserMedia, so no microphone";
  }
  return null;
}

/**
 * Resolve once `context` is running, or throw a `NotAllowedError`. Outside a
 * gesture `resume()` may wait for one, so it is raced against a timeout.
 */
async function ensureRunning(context) {
  if (context.state !== "running") {
    await Promise.race([
      context.resume().catch(() => {}),
      new Promise((resolve) => window.setTimeout(resolve, RESUME_TIMEOUT_MS)),
    ]);
  }
  if (context.state !== "running") {
    throw new DOMException("the browser kept the audio suspended", "NotAllowedError");
  }
}

/** Root-mean-square of a PCM16 frame, as a 0..1 meter position. */
function frameLevel(buffer) {
  const view = new DataView(buffer);
  const samples = Math.floor(buffer.byteLength / 2);
  if (!samples) return 0;
  // Every fourth sample is plenty for a meter.
  let sum = 0;
  let taken = 0;
  for (let index = 0; index < samples; index += 4) {
    const value = view.getInt16(index * 2, true) / 32768;
    sum += value * value;
    taken += 1;
  }
  const rms = Math.sqrt(sum / Math.max(1, taken));
  const db = 20 * Math.log10(Math.max(rms, 1e-7));
  return Math.max(0, Math.min(1, (db + 60) / 60));
}

/** The microphone. Speech during the socket handshake is queued, not lost. */
function micSource() {
  let stream = null;
  let node = null;
  let halted = false;
  return {
    mode: "mic",
    name: "microphone",
    element: null,
    async open(context, sink) {
      const opened = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          // Off: AGC drifts the gain, so one voice would reach the speaker
          // encoder at a level set by whoever spoke last.
          autoGainControl: false,
        },
      });
      if (halted) {
        // Stopped while the permission prompt was still up.
        for (const track of opened.getTracks()) track.stop();
        return;
      }
      stream = opened;
      node = context.createMediaStreamSource(stream);
      node.connect(sink);
    },
    async play() {},
    halt() {
      halted = true;
      try {
        node?.disconnect();
      } catch (_) { /* already detached */ }
      for (const track of stream?.getTracks() || []) track.stop();
    },
  };
}

/**
 * A file, played out loud and streamed as it plays. It has its own `Audio`
 * element, not the review player: an element feeds only one source node, and
 * a seek would send audio out of order. It feeds the worklet only while
 * playing, so a pause is a pause on the wire too.
 */
function fileSource(file) {
  const url = URL.createObjectURL(file);
  const element = new Audio();
  element.preload = "auto";
  element.src = url;
  let node = null;
  let sink = null;
  let feeding = false;
  const feed = (on) => {
    if (!node || on === feeding) return;
    feeding = on;
    try {
      if (on) node.connect(sink);
      else node.disconnect(sink);
    } catch (_) { /* the graph is going away */ }
  };
  element.addEventListener("play", () => feed(true));
  element.addEventListener("pause", () => feed(false));
  return {
    mode: "file",
    name: file.name || "audio",
    element,
    async open(context, target) {
      sink = target;
      node = context.createMediaElementSource(element);
      node.connect(context.destination); // audible
    },
    async play() {
      // Attached before `play()`, so no audio slips past.
      feed(true);
      try {
        await element.play();
      } catch (err) {
        feed(false);
        throw err;
      }
    },
    halt() {
      feed(false);
      element.pause();
      element.removeAttribute("src");
      element.load();
      URL.revokeObjectURL(url);
    },
  };
}

/**
 * Stream `source` until it ends, is stopped, or fails. Stop-and-wait: one
 * chunk in flight, released by its `chunk_accepted`. A queue that reaches its
 * ceiling stops the session rather than drop audio.
 */
async function startCapture(source, context) {
  const live = {
    socket: null,
    node: null,
    queue: [],
    sequence: 0,
    inFlight: false,
    started: false, // `session_started` has arrived
    sealing: false, // no more audio is taken in
    flushed: false, // the worklet has drained its resampler
    sentEos: false,
    released: false,
    flushTimer: 0,
    level: 0,
    /** Samples taken in at `LIVE_SAMPLE_RATE`: the stream's clock. */
    captured: 0,
    /** Every frame the server accepted, in order, for playback at the end. */
    sent: [],
  };
  transport = live;

  /** Tear the capture down, once. `hangUp` also cancels and closes the socket. */
  const release = ({ hangUp }) => {
    if (live.released) return;
    live.released = true;
    if (transport === live) transport = null;
    window.clearTimeout(live.flushTimer);
    source.halt();
    try {
      live.node?.port.close();
      live.node?.disconnect();
    } catch (_) { /* the graph is being torn down either way */ }
    context.close().catch(() => {});
    live.queue.length = 0;
    $("level").style.width = "0%";
    const socket = live.socket;
    if (hangUp && socket && socket.readyState <= WebSocket.OPEN) {
      try {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "cancel" }));
        }
        socket.close();
      } catch (_) { /* closing is all that was left to do */ }
    }
  };

  const fail = (message) => {
    if (live.released) return;
    release({ hangUp: true });
    onError(message);
  };
  live.fail = fail;

  const sendNext = () => {
    if (live.released || live.inFlight || live.queue.length === 0) return;
    if (live.socket?.readyState !== WebSocket.OPEN) return;
    const frame = live.queue[0];
    try {
      live.socket.send(JSON.stringify({
        type: "chunk",
        sequence: live.sequence,
        byte_length: frame.byteLength,
      }));
      live.socket.send(frame);
    } catch (err) {
      fail(`the live socket refused a frame: ${err.message}`);
      return;
    }
    live.sequence += 1;
    live.inFlight = true;
  };

  /** `eos` once the worklet is drained and every frame has been accepted. */
  const sealWhenDrained = () => {
    if (live.released || live.sentEos || !live.flushed) return;
    if (live.inFlight || live.queue.length) return;
    if (live.socket?.readyState !== WebSocket.OPEN) return;
    try {
      live.socket.send(JSON.stringify({ type: "eos" }));
    } catch (err) {
      fail(`the live socket refused the end of stream: ${err.message}`);
      return;
    }
    live.sentEos = true;
  };

  const onFrame = (buffer) => {
    if (live.released || live.sentEos) return;
    if (live.queue.length >= MAX_PENDING_FRAMES) {
      // Never silently drop captured audio. Stop, and say why.
      fail(BACKLOG_MESSAGE);
      return;
    }
    live.level = Math.max(frameLevel(buffer), live.level * 0.86);
    $("level").style.width = `${(live.level * 100).toFixed(0)}%`;
    // Counted on intake: a queued frame reaches the server or the session fails.
    live.captured += Math.floor(buffer.byteLength / 2);
    const position = live.captured / LIVE_SAMPLE_RATE;
    renderTime(position);
    const total = source.element?.duration;
    if (Number.isFinite(total) && total > 0) {
      $("fill").style.width = `${Math.min(100, (position / total) * 100).toFixed(1)}%`;
    }
    live.queue.push(buffer);
    sendNext();
  };

  const onAccepted = (sequence) => {
    if (!live.inFlight || sequence !== live.sequence - 1) return;
    live.sent.push(live.queue.shift());
    live.inFlight = false;
    sendNext();
    sealWhenDrained();
  };

  const markFlushed = () => {
    window.clearTimeout(live.flushTimer);
    live.flushed = true;
    sealWhenDrained();
  };

  /** Stop taking audio in, drain the worklet, then `eos`: the tail is kept. */
  const seal = () => {
    if (live.released || live.sealing) return;
    live.sealing = true;
    $("stop").disabled = true;
    $("resume").hidden = true;
    say("transcribing the last few seconds…");
    source.halt();
    if (!live.node) {
      markFlushed();
      return;
    }
    live.node.port.postMessage({ type: "flush" });
    // A worklet that never answers must not hold the session open for good.
    live.flushTimer = window.setTimeout(markFlushed, FLUSH_TIMEOUT_MS);
  };

  const offerResume = (message) => {
    if (live.released || live.sealing) return;
    $("resume").hidden = false;
    say(message, "warn");
  };

  /** Start the audio. It needs a running context, which may need a click. */
  const begin = async () => {
    try {
      await ensureRunning(context);
      await source.play();
    } catch (err) {
      if (live.released || live.sealing) return;
      if (err?.name === "NotAllowedError") {
        offerResume("the browser wants a click before it plays audio: press Play");
      } else {
        fail(`${source.name} could not be played: ${err?.message || err}`);
      }
      return;
    }
    if (live.released || live.sealing) return;
    $("resume").hidden = true;
    say(source.mode === "mic" ? "listening" : "playing");
  };

  // Called from a click, so a context the browser held back can start now.
  live.resume = () => {
    if (live.released || live.sealing || !live.started) return;
    begin();
  };

  live.stop = () => {
    if (live.released) return;
    if (live.sequence === 0 && live.queue.length === 0) {
      // Nothing reached the server, so there is nothing to finalize.
      release({ hangUp: true });
      onAbandoned();
      return;
    }
    seal();
  };

  // The source, and a file's length.
  $("note").textContent = source.name;
  const element = source.element;
  if (element) {
    element.addEventListener("durationchange", () => {
      if (live.released) return;
      const total = element.duration;
      if (Number.isFinite(total) && total > 0) {
        $("note").textContent = `${source.name} · ${formatClock(total)}`;
      }
    });
    element.addEventListener("ended", seal);
    element.addEventListener("pause", () => {
      // A pause the page did not ask for, such as a media key.
      if (!element.ended) offerResume("paused: press Play to continue");
    });
    element.addEventListener("error", () => {
      if (live.released || live.sealing) return;
      fail(`${source.name} could not be decoded by this browser`);
    });
  }

  try {
    await context.audioWorklet.addModule("/assets/pcm-worklet.js");
    if (live.released) return;
    live.node = new AudioWorkletNode(context, "pcm16-streamer", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      // Down-mixed here, so a stereo file is not heard as its left channel.
      channelCount: 1,
      channelCountMode: "explicit",
      channelInterpretation: "speakers",
      processorOptions: {
        targetSampleRate: LIVE_SAMPLE_RATE,
        chunkSamples: LIVE_CHUNK_SAMPLES,
      },
    });
    live.node.port.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) onFrame(event.data);
      else if (event.data?.type === "flush_complete") markFlushed();
    };
    // Keeps the graph pulling; the processor outputs silence.
    live.node.connect(context.destination);
  } catch (err) {
    fail(`the capture graph could not be built: ${err.message}`);
    return;
  }

  try {
    await source.open(context, live.node);
  } catch (err) {
    fail(source.mode === "mic"
      ? `the microphone could not be opened: ${err.message}`
      : `${source.name} could not be opened: ${err.message}`);
    return;
  }
  if (live.released) return;

  let socket;
  try {
    socket = new WebSocket(buildLiveWebSocketUrl());
  } catch (err) {
    fail(`the live socket could not be opened: ${err.message}`);
    return;
  }
  live.socket = socket;
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    if (live.released) return;
    try {
      // No `step_sec`: the server owns the tick schedule.
      socket.send(JSON.stringify({
        type: "start",
        sample_rate: LIVE_SAMPLE_RATE,
        channels: 1,
        format: "pcm_s16le",
      }));
    } catch (err) {
      fail(`the live socket refused the start message: ${err.message}`);
      return;
    }
    // Anything the microphone heard while the socket was connecting.
    sendNext();
  };

  socket.onmessage = (event) => {
    if (live.released || typeof event.data !== "string") return;
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (err) {
      fail(`the server sent a frame the page could not read: ${err.message}`);
      return;
    }
    switch (message.type) {
      case "session_started":
        live.started = true;
        state.status = "streaming";
        begin();
        return;
      case "chunk_accepted":
        onAccepted(finiteNumber(message.sequence, -1));
        return;
      case "tick":
        onTick(message);
        return;
      case "done":
        // The server closes the socket itself once its worker is finished.
        release({ hangUp: false });
        onDone(message, live.sent);
        live.sent = [];
        return;
      case "error":
        release({ hangUp: true });
        onError(message.message);
        return;
      default:
        return;
    }
  };

  socket.onclose = (event) => {
    if (live.released) return;
    fail(event.code === 1008
      ? "the server refused the live session (policy violation)"
      : `the live socket closed before the transcript finished (code ${event.code})`);
  };
}

/**
 * Start a session from a click or a drop. The `AudioContext` is created
 * synchronously inside the gesture; after an `await` it may start suspended.
 */
function launch(source, announce) {
  let context;
  try {
    context = new AudioContext();
  } catch (err) {
    source.halt();
    say(`the audio graph could not be created: ${err.message}`, "err");
    return;
  }
  context.resume().catch(() => {});
  beginSession(source.mode, source.name);
  say(announce);
  startCapture(source, context).catch((err) => {
    // An unexpected error still ends the session.
    const message = `the session failed: ${err?.message || err}`;
    if (transport?.fail) transport.fail(message);
    else if (state.running) onError(message);
  });
}

function startFile(file) {
  if (state.running) {
    say("a session is already running; stop it first", "warn");
    return;
  }
  if (captureBlocker) {
    say(captureBlocker, "err");
    return;
  }
  launch(fileSource(file), "connecting…");
}

function startMic() {
  if (state.running) return;
  const blocker = captureBlocker || micBlocker;
  if (blocker) {
    say(blocker, "err");
    return;
  }
  launch(micSource(), "asking for the microphone…");
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function download(text, name, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoked on the next turn, once the browser has taken the blob.
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

$("pick").addEventListener("click", () => $("file").click());

$("file").addEventListener("change", () => {
  const file = $("file").files[0];
  // Cleared, so choosing the same file again still fires ``change``.
  $("file").value = "";
  if (file) startFile(file);
});

$("mic").addEventListener("click", startMic);

// Stop seals: the server publishes the tail, so nothing heard is lost.
$("stop").addEventListener("click", () => transport?.stop?.());

$("resume").addEventListener("click", () => transport?.resume?.());

// Drop anywhere; without `preventDefault` the browser would open the file.
let dragDepth = 0;
const carriesFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
window.addEventListener("dragenter", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth += 1;
  $("drop").hidden = state.running;
});
window.addEventListener("dragover", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = state.running ? "none" : "copy";
});
window.addEventListener("dragleave", (event) => {
  if (!carriesFiles(event)) return;
  // Leaving a child element fires too; only the last leave hides the overlay.
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) $("drop").hidden = true;
});
window.addEventListener("drop", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  $("drop").hidden = true;
  const file = event.dataTransfer.files[0];
  if (file) startFile(file);
});

const player = $("player");
for (const name of ["timeupdate", "seeked", "play", "pause"]) {
  player.addEventListener(name, renderTranscript);
}
player.addEventListener("error", () => {
  if (!player.currentSrc) return;
  $("player-note").textContent = "the audio could not be played back";
});

$("srt").addEventListener("click", () => {
  download(buildSrt(state.published), downloadName(state.audioName, "srt"), "text/plain");
});

$("json").addEventListener("click", () => {
  download(
    buildJsonExport(state.published, { audioName: state.audioName, status: state.status }),
    downloadName(state.audioName, "json"),
    "application/json",
  );
});

// Say up front what this browser cannot do.
captureBlocker = captureProblem();
micBlocker = microphoneProblem();
endSession();
renderTranscript();
