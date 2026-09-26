"""Canonical append-only online session for streaming speaker ASR."""

from __future__ import annotations

from pathlib import Path
import resource
from typing import Any, Mapping

from ..core.decode.commit import IncrementalCommitter
from ..core.decode.engine import IncrementalEngine, WindowSpec
from ..core.decode.moss_adapter import MossWindowDecoder
from ..core.decode.validation import require_publishable
from .causal import CausalSegmentStore
from . import pipeline as _pipeline


class StreamingSpeakerASRSession:
    """Own one decoder, identity filter, and publication state until EOS."""

    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: str | Path,
        *,
        audio_source: Any = None,
        decoder: Any = None,
        embedding_store: Any = None,
        attribution_observer: Any = None,
    ) -> None:
        normalized = _pipeline.validate_config(
            config,
            require_embedding=True,
            require_association=True,
        )
        self.config = normalized
        self.sample = _pipeline._mapping(normalized, "sample")
        self.stream = _pipeline._mapping(normalized, "stream")
        self.inference = _pipeline._mapping(normalized, "inference")
        self.association = _pipeline._mapping(normalized, "association")
        self.embedding_config = _pipeline._mapping(normalized, "embedding")
        speaker_mapping = normalized.get("speaker_mapping", {})
        if not isinstance(speaker_mapping, Mapping):
            raise ValueError("config.speaker_mapping must be a mapping")
        self.speaker_mapping = speaker_mapping
        self.destination = Path(output_dir).expanduser().resolve()
        self.destination.mkdir(parents=True, exist_ok=True)
        self.audio_source = audio_source
        self.embedding_store = embedding_store or self._build_embedding_store()
        self.attribution_observer = attribution_observer
        self._speaker_binding_runtime = _pipeline.SpeakerBindingRuntime()
        self.postcommit_assigner = self._build_assigner()
        self.committer = self._build_committer()
        self.decoder = decoder or MossWindowDecoder(
            audio_path=Path(str(self.sample["audio_path"])),
            endpoint=str(normalized["endpoint"]),
            model=str(normalized["served_model_name"]),
            prompt=str(self.inference.get("prompt", "")),
            max_new_tokens=int(
                self.inference.get("window_max_new_tokens", 4096)
            ),
            short_window_max_new_tokens=int(
                self.inference.get("short_window_max_new_tokens", 512)
            ),
            short_window_threshold_sec=float(
                self.inference.get("short_window_threshold_sec", 9.0)
            ),
            timeout_sec=_pipeline._decoder_timeout_sec(
                self.inference,
                self.stream,
            ),
            sse_soft_deadline_sec=float(
                self.inference.get("sse_soft_deadline_sec", 0.0)
            ),
            causal_frontier_request_ids=bool(
                self.inference.get("causal_frontier_request_ids", False)
            ),
            causal_frontier_time_warp=bool(
                self.inference.get("causal_frontier_time_warp", False)
            ),
            attribution_observer=attribution_observer,
            audio_source=audio_source,
        )
        self.engine = IncrementalEngine(
            duration_sec=float(self.sample["duration_sec"]),
            step_sec=float(self.stream["step_sec"]),
            context_sec=float(self.stream["context_sec"]),
            right_context_sec=float(self.stream["right_context_sec"]),
            decode=self.decoder,
            map_segments=lambda segments, _window: list(segments),
            after_update=self._after_update,
            committer=self.committer,
            max_wall_latency_sec=(
                None
                if self.stream.get("max_wall_latency_sec") in (None, "")
                else float(self.stream["max_wall_latency_sec"])
            ),
            continue_on_decode_error=bool(
                self.inference.get("continue_on_decode_error", True)
            ),
            recoverable_decode_error_filter=(
                _pipeline._is_recoverable_decode_error
            ),
            sample_metadata={
                "audio_name": self.sample.get(
                    "audio_name",
                    self.sample.get("id", "sample"),
                ),
                "audio_path": self.sample.get("audio_path", ""),
                "language": self.sample.get("language"),
            },
            attribution_observer=attribution_observer,
        )
        self.engine_session = self.engine.start_session()
        self._processed_ticks = 0
        self._last_tick = 0.0
        self._finalized = False

    def _build_embedding_store(self) -> CausalSegmentStore:
        return CausalSegmentStore(
            audio_path=Path(str(self.sample["audio_path"])),
            speakerlab_root=Path(str(self.embedding_config.get("speakerlab_root") or self.embedding_config["repo_root"])),
            model_dir=Path(str(self.embedding_config["model_dir"])),
            model_name=str(
                self.embedding_config.get("model_name", "eres2netv2")
            ),
            device=str(self.embedding_config.get("device", "cuda:0")),
            batch_size=int(self.embedding_config.get("batch_size", 64)),
            chunk_duration_sec=float(
                self.embedding_config.get("chunk_duration_sec", 1.5)
            ),
            chunk_step_sec=float(
                self.embedding_config.get("chunk_step_sec", 0.75)
            ),
            checkpoint_name=str(
                self.embedding_config.get(
                    "checkpoint_name",
                    "pretrained_eres2netv2w24s4ep4.ckpt",
                )
            ),
            audio_source=self.audio_source,
        )

    def _build_assigner(self) -> _pipeline.CausalSpeakerBinder:
        assigner = _pipeline.build_speaker_binder(
            sample=self.sample,
            association=self.association,
            speaker_mapping=self.speaker_mapping,
            embedding_config=self.embedding_config,
            embedding_store=self.embedding_store,
            runtime=self._speaker_binding_runtime,
        )
        if assigner is None:
            raise RuntimeError("causal speaker binding requires an embedding store")
        return assigner

    def _build_committer(self) -> IncrementalCommitter:
        return _pipeline.build_speaker_binding_committer(
            stream=self.stream,
            association=self.association,
            assigner=self.postcommit_assigner,
            runtime=self._speaker_binding_runtime,
        )

    def _after_update(
        self,
        _window: Any,
        _decoded: Any,
        _mapped: Any,
        _update: Any,
    ) -> Mapping[str, Any] | None:
        return self._speaker_binding_runtime.event_metadata()

    def process_tick(self, end_sec: float, *, final: bool) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("online session is finalized")
        end = float(end_sec)
        if end <= self._last_tick + 1e-9:
            raise ValueError("online tick replay is not allowed")
        if self.audio_source is not None:
            available = float(self.audio_source.available_until_sec)
            if end > available + 1e-9:
                raise ValueError("online tick exceeds available causal audio")
        window = WindowSpec(
            tick=end,
            start=max(0.0, end - float(self.stream["context_sec"])),
            end=end,
        )
        row = self.engine_session.process(window, final=bool(final))
        self._processed_ticks += 1
        self._last_tick = end
        return row

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("online session is finalized")
        result = self.engine_session.finalize(
            expected_ticks=self._processed_ticks
        )
        local_committed = tuple(result.committed)
        final_segments = list(
            self.postcommit_assigner.view(local_committed)
        )
        result.committed[:] = final_segments
        validation = result.validation
        raw_count = len(local_committed)
        final_count = len(final_segments)
        validation["raw_committed_segment_count"] = raw_count
        validation["committed_segment_count"] = final_count
        observed_material = bool(
            validation.get("observed_material_content", False)
        )
        validation["final_content_gate_pass"] = bool(
            not observed_material or final_count > 0
        )
        decision_events = [
            dict(event)
            for row in result.rows
            for event in row.get("speaker_decision_events", [])
        ]
        resolution_events = [
            dict(event)
            for row in result.rows
            for event in row.get("speaker_resolution_events", [])
        ]
        assignment_count = sum(
            1
            for row in result.rows
            for _item in row.get("speaker_assignments", [])
        )
        max_wait = max(
            (int(event.get("wait_ticks", 0)) for event in decision_events),
            default=0,
        )
        published_count = self.postcommit_assigner.published_segment_count
        pending_speaker_count = (
            self.postcommit_assigner.pending_speaker_count()
        )
        pending_segment_count = len(
            self.postcommit_assigner.pending_segments()
        )
        source_keys = [
            str(event.get("source_key", "")) for event in decision_events
        ]
        counters = self.postcommit_assigner.core_counters
        validation.update(
            {
                "active_identity_decision_count": len(resolution_events),
                "final_authoritative_identity_digest": (
                    self.postcommit_assigner.final_authoritative_identity_digest()
                ),
                "process_peak_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "postcommit_speaker_assignment_count": assignment_count,
                "speaker_decision_count": len(decision_events),
                "speaker_decision_mode_counts": {
                    mode: sum(
                        1
                        for event in decision_events
                        if str(event.get("mode", "unknown")) == mode
                    )
                    for mode in sorted(
                        {
                            str(event.get("mode", "unknown"))
                            for event in decision_events
                        }
                    )
                },
                "max_speaker_wait_ticks": max_wait,
                "speaker_wait_tick_gate_pass": (
                    max_wait
                    <= self.postcommit_assigner.best_effort_after_ticks
                ),
                "published_segment_count": published_count,
                "committed_equals_published": raw_count == published_count,
                "pending_speaker_count": pending_speaker_count,
                "pending_segment_count": pending_segment_count,
                "duplicate_publication_count": (
                    len(source_keys) - len(set(source_keys))
                ),
                "silent_speaker_fallback_count": max(
                    0,
                    published_count - len(decision_events),
                ),
                "eos_novelty_count": sum(
                    1
                    for event in decision_events
                    if bool(event.get("eos"))
                    and str(event.get("mode")) == "deadline"
                    and str(event.get("status")) == "new"
                    and str(event.get("evidence")) != "bootstrap"
                ),
                "postcommit_embedding_call_count": (
                    self.postcommit_assigner.embedding_call_count
                ),
                "postcommit_embedding_available_count": (
                    self.postcommit_assigner.embedding_available_count
                ),
                "postcommit_embedding_counters": counters,
                "speaker_policy": str(
                    self.association.get(
                        "policy",
                        _pipeline.LEGO_POLICY,
                    )
                ),
                "postprocess_gate_pass": True,
            }
        )
        for name, value in counters.items():
            validation[f"postcommit_embedding_{name}"] = int(value)
        speaker_gate = bool(
            pending_speaker_count == 0
            and pending_segment_count == 0
            and self.postcommit_assigner.is_fully_assigned(result.committed)
            and validation["committed_equals_published"]
            and validation["speaker_wait_tick_gate_pass"]
        )
        validation["speaker_assignment_gate_pass"] = speaker_gate
        validation["status"] = (
            "passed"
            if validation["status"] == "passed"
            and validation["final_content_gate_pass"]
            and speaker_gate
            else "failed"
        )
        self.engine._write_jsonl(
            self.destination / "results" / "ticks.jsonl",
            result.rows,
        )
        _pipeline._write_json(
            self.destination / "stream_validation.json",
            validation,
        )
        _pipeline._write_json(
            self.destination / "hypothesis.json",
            [
                {
                    "audio_name": self.sample.get(
                        "audio_name",
                        self.sample.get("id", "sample"),
                    ),
                    "audio_path": self.sample.get("audio_path", ""),
                    "segments": [item.to_dict() for item in result.committed],
                }
            ],
        )
        require_publishable(self.destination, mode="incremental_asr")
        association_summary = _pipeline._materialize_causal_artifact(
            self.destination
        )
        self._finalized = True
        stream_result = {
            "mode": "incremental_asr",
            "status": validation["status"],
            "results": str(
                self.destination / "results" / "ticks.jsonl"
            ),
            "validation": str(
                self.destination / "stream_validation.json"
            ),
            "hypothesis": str(self.destination / "hypothesis.json"),
            "committed_segments": final_count,
            "speaker_count": len(
                {segment.speaker for segment in result.committed}
            ),
            "confirmed_speaker_count": (
                self.postcommit_assigner.confirmed_speaker_count()
            ),
            "pending_speaker_count": pending_speaker_count,
            "published_segment_count": published_count,
            "pending_segment_count": pending_segment_count,
            "duplicate_publication_count": validation[
                "duplicate_publication_count"
            ],
            "silent_speaker_fallback_count": validation[
                "silent_speaker_fallback_count"
            ],
            "eos_novelty_count": validation["eos_novelty_count"],
            "max_speaker_wait_ticks": max_wait,
            "speaker_policy": validation["speaker_policy"],
        }
        return {
            "mode": "global_embedding",
            "stream": stream_result,
            "embedding": self.embedding_store.summary(),
            "association": association_summary,
            "association_mode": "causal_embedding",
            "speaker_policy": validation["speaker_policy"],
        }

    def fail(self, exc: Exception) -> None:
        self.engine_session.fail(
            exc,
            expected_ticks=max(1, self._processed_ticks),
        )


__all__ = ["StreamingSpeakerASRSession"]
