/**
 * Capture-side audio, loaded with `AudioWorklet.addModule`: resample the
 * context's rate (usually 44.1 or 48 kHz) to 16 kHz, cut PCM16 frames, and
 * post them as transferred `ArrayBuffer`s. Incremental, block by block, since
 * `process()` must return within one render quantum.
 */

/** Streaming linear resampler: continuous across calls, not per-block. */
export class StreamingLinearResampler {
  constructor(sourceSampleRate, targetSampleRate) {
    this.sourceSampleRate = Number(sourceSampleRate);
    this.targetSampleRate = Number(targetSampleRate);
    if (!(this.sourceSampleRate > 0) || !(this.targetSampleRate > 0)) {
      throw new RangeError("sample rates must be positive");
    }
    // Input frames per output frame, usually fractional.
    this._step = this.sourceSampleRate / this.targetSampleRate;
    this._position = 0;
    this._buffer = new Float32Array(0);
  }

  /** Resample one block; returns the frames that are now complete. */
  push(values) {
    const input = values instanceof Float32Array ? values : Float32Array.from(values || []);
    if (!input.length) return new Float32Array(0);
    // The last block's tail: a frame at the boundary needs its right neighbour.
    const joined = new Float32Array(this._buffer.length + input.length);
    joined.set(this._buffer);
    joined.set(input, this._buffer.length);
    const output = [];
    while (this._position + 1 < joined.length) {
      const leftIndex = Math.floor(this._position);
      const fraction = this._position - leftIndex;
      const left = joined[leftIndex];
      const right = joined[leftIndex + 1];
      output.push(left + (right - left) * fraction);
      this._position += this._step;
    }
    // Keep what the next call still interpolates from.
    const discard = Math.min(Math.floor(this._position), joined.length);
    this._buffer = joined.slice(discard);
    this._position -= discard;
    return Float32Array.from(output);
  }

  /** Drain the tail at end of stream, holding the last sample for the edge. */
  flush() {
    if (!this._buffer.length) return new Float32Array(0);
    const output = [];
    while (this._position < this._buffer.length) {
      const leftIndex = Math.floor(this._position);
      const rightIndex = Math.min(leftIndex + 1, this._buffer.length - 1);
      const fraction = this._position - leftIndex;
      const left = this._buffer[leftIndex];
      const right = this._buffer[rightIndex];
      output.push(left + (right - left) * fraction);
      this._position += this._step;
    }
    this._buffer = new Float32Array(0);
    this._position = 0;
    return Float32Array.from(output);
  }
}


/** Accumulates float samples into fixed-size little-endian PCM16 frames. */
export class Pcm16Chunker {
  constructor(chunkSamples = 1600) {
    this.chunkSamples = Math.trunc(Number(chunkSamples));
    if (this.chunkSamples <= 0) throw new RangeError("chunkSamples must be positive");
    this._pending = [];
  }

  /** Clamp, scale to the int16 range, and serialise as little-endian. */
  static encode(values) {
    const buffer = new ArrayBuffer(values.length * 2);
    const view = new DataView(buffer);
    for (let index = 0; index < values.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, Number(values[index]) || 0));
      // Asymmetric, so -32768 is reachable; the server decodes with /32768.
      const pcm = sample < 0
        ? Math.round(sample * 32768)
        : Math.round(sample * 32767);
      view.setInt16(index * 2, pcm, true);
    }
    return buffer;
  }

  /** Absorb samples; return every whole frame that is now available. */
  push(values) {
    for (const value of values) this._pending.push(value);
    const frames = [];
    while (this._pending.length >= this.chunkSamples) {
      frames.push(Pcm16Chunker.encode(this._pending.splice(0, this.chunkSamples)));
    }
    return frames;
  }

  /** The short final frame at end of stream, or `null` if it divides evenly. */
  flush() {
    if (!this._pending.length) return null;
    return Pcm16Chunker.encode(this._pending.splice(0));
  }
}


// Only defined in a worklet scope; the fallback keeps the module importable.
const WorkletBase = globalThis.AudioWorkletProcessor || class {};


/**
 * Resample, chunk, and post: `ArrayBuffer` frames, then one
 * `{type: "flush_complete"}` once a requested flush has drained.
 */
class Pcm16StreamerProcessor extends WorkletBase {
  constructor(options = {}) {
    super();
    const processorOptions = options.processorOptions || {};
    // The context's own rate; the option is a fallback.
    const sourceRate = Number(globalThis.sampleRate || processorOptions.sourceSampleRate || 48000);
    const targetRate = Number(processorOptions.targetSampleRate || 16000);
    this._resampler = new StreamingLinearResampler(sourceRate, targetRate);
    // 0.1 s at 16 kHz, well under the server's one-second frame limit.
    this._chunker = new Pcm16Chunker(processorOptions.chunkSamples || 1600);
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") {
        this._emit(this._chunker.push(this._resampler.flush()));
        const residual = this._chunker.flush();
        if (residual) this.port.postMessage(residual, [residual]);
        this.port.postMessage({type: "flush_complete"});
      }
    };
  }

  _emit(frames) {
    for (const frame of frames) this.port.postMessage(frame, [frame]);
  }

  process(inputs, outputs) {
    const channel = inputs[0]?.[0];
    if (channel?.length) this._emit(this._chunker.push(this._resampler.push(channel)));
    // Silence out: the microphone is never played back.
    for (const output of outputs[0] || []) output.fill(0);
    return true;
  }
}


if (typeof globalThis.registerProcessor === "function") {
  globalThis.registerProcessor("pcm16-streamer", Pcm16StreamerProcessor);
}
