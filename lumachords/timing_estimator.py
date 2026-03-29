from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TimeSignatureCandidate:
    time_signature: tuple[int, int]
    score: float


@dataclass(frozen=True)
class TimingEstimationResult:
    bpm: float | None
    time_signature: tuple[int, int] | None
    confidence: float
    candidates: list[TimeSignatureCandidate]


class TimingEstimator:
    DEFAULT_TIME_SIGNATURE_CANDIDATES = (
        (2, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (6, 8),
        (9, 8),
        (12, 8),
    )

    def __init__(
        self,
        min_bpm: float = 40.0,
        max_bpm: float = 220.0,
        bpm_step: float = 0.5,
        onset_merge_tolerance_secs: float = 0.03,
        min_required_onsets: int = 6,
    ):
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.bpm_step = float(bpm_step)
        self.onset_merge_tolerance_secs = float(onset_merge_tolerance_secs)
        self.min_required_onsets = int(min_required_onsets)

    def estimate(self, event_pairs_secs: np.ndarray) -> TimingEstimationResult: # DT_PAIR_SECS
        event_pairs_secs = event_pairs_secs[event_pairs_secs["staff"] == 1].copy()
        crop_secs = event_pairs_secs["on_secs"].min()
        event_pairs_secs["on_secs"] -= crop_secs
        event_pairs_secs["off_secs"] -= crop_secs

        onsets, weights = self._extract_onsets(event_pairs_secs)
        if onsets.size < self.min_required_onsets:
            return TimingEstimationResult(bpm=None, time_signature=None, confidence=0.0, candidates=[])

        bpm, bpm_score = self._estimate_bpm(onsets, weights)
        if bpm is None:
            return TimingEstimationResult(bpm=None, time_signature=None, confidence=0.0, candidates=[])

        candidates = self._estimate_time_signature_candidates(onsets, weights, bpm)
        if not candidates:
            return TimingEstimationResult(bpm=bpm, time_signature=None, confidence=min(1.0, max(0.0, bpm_score)), candidates=[])

        top = candidates[0]
        second = candidates[1].score if len(candidates) > 1 else 0.0
        ts_conf = max(0.0, top.score - second)
        confidence = float(max(0.0, min(1.0, (0.6 * bpm_score) + (0.4 * ts_conf))))
        return TimingEstimationResult(
            bpm=float(round(bpm, 2)),
            time_signature=top.time_signature,
            confidence=confidence,
            candidates=candidates,
        )

    def _extract_onsets(self, event_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]: # DT_PAIR_SECS
        if not len(event_pairs):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        rows = []
        # (hands_idx, staff, midi_num, on_secs, off_secs)
        for on_secs, off_secs in event_pairs[["on_secs", "off_secs"]]:
            try:
                on = float(on_secs)
                off = float(off_secs)
            except Exception:
                continue
            if not math.isfinite(on) or not math.isfinite(off):
                continue
            if off <= on:
                continue
            dur = max(0.04, min(2.0, off - on))
            # Short notes still contribute, long notes get slightly more emphasis.
            weight = 1.0 + 0.25 * math.log1p(dur)
            rows.append((on, weight))

        if not rows:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        rows.sort(key=lambda x: x[0])
        merged_onsets: list[float] = []
        merged_weights: list[float] = []
        cur_t, cur_w = rows[0]
        tol = self.onset_merge_tolerance_secs
        for t, w in rows[1:]:
            if abs(t - cur_t) <= tol:
                cur_w += w
            else:
                merged_onsets.append(cur_t)
                merged_weights.append(cur_w)
                cur_t, cur_w = t, w
        merged_onsets.append(cur_t)
        merged_weights.append(cur_w)

        return np.asarray(merged_onsets, dtype=np.float64), np.asarray(merged_weights, dtype=np.float64)

    def _estimate_bpm(self, onsets: np.ndarray, weights: np.ndarray) -> tuple[float | None, float]:
        bpm_grid = np.arange(self.min_bpm, self.max_bpm + self.bpm_step * 0.5, self.bpm_step, dtype=np.float64)
        if bpm_grid.size == 0:
            return None, 0.0

        scores = np.zeros_like(bpm_grid)
        for i, bpm in enumerate(bpm_grid):
            beat_period = 60.0 / bpm
            s_main = self._beat_alignment_score(onsets, weights, beat_period)
            s_half = self._beat_alignment_score(onsets, weights, beat_period * 2.0)
            s_double = self._beat_alignment_score(onsets, weights, beat_period * 0.5)
            # Harmonics reduce half/double-time ambiguity.
            scores[i] = 0.65 * s_main + 0.2 * s_half + 0.15 * s_double

        best_i = int(np.argmax(scores))
        best_bpm = float(bpm_grid[best_i])
        best_score = float(scores[best_i])
        mean_score = float(np.mean(scores))
        bpm_conf = max(0.0, min(1.0, (best_score - mean_score) * 2.0))
        return best_bpm, bpm_conf

    def _beat_alignment_score(self, onsets: np.ndarray, weights: np.ndarray, period: float) -> float:
        if period <= 0:
            return 0.0
        phase = np.mod(onsets, period)
        dist = np.minimum(phase, period - phase)
        tol = max(0.02, 0.14 * period)
        fit = np.exp(-((dist / tol) ** 2))
        weighted = fit * weights
        denom = float(np.sum(weights)) + 1e-12
        return float(np.sum(weighted) / denom)

    def _estimate_time_signature_candidates(
        self, onsets: np.ndarray, weights: np.ndarray, bpm: float
    ) -> list[TimeSignatureCandidate]:
        beat_period = 60.0 / bpm
        meter_scores = {beats: self._meter_score(onsets, weights, beat_period, beats) for beats in (2, 3, 4, 5)}
        binary_subdiv_score = self._subdivision_alignment_score(onsets, weights, beat_period, 2)
        ternary_subdiv_score = self._subdivision_alignment_score(onsets, weights, beat_period, 3)

        candidates: list[TimeSignatureCandidate] = []
        for ts in self.DEFAULT_TIME_SIGNATURE_CANDIDATES:
            beats, is_compound = self._signature_to_tactus_beats(ts)
            score = meter_scores.get(beats, 0.0)
            if is_compound:
                score += 0.15 * (ternary_subdiv_score - binary_subdiv_score)
            else:
                score += 0.10 * (binary_subdiv_score - ternary_subdiv_score)

            # Small prior toward common signatures.
            if ts == (4, 4):
                score += 0.03
            elif ts in ((3, 4), (6, 8)):
                score += 0.015

            candidates.append(TimeSignatureCandidate(time_signature=ts, score=float(score)))

        candidates.sort(key=lambda c: c.score, reverse=True)
        if not candidates:
            return candidates

        top_score = candidates[0].score
        if top_score <= 1e-8:
            return candidates
        return [
            TimeSignatureCandidate(time_signature=c.time_signature, score=float(max(0.0, c.score / top_score)))
            for c in candidates
        ]

    def _meter_score(self, onsets: np.ndarray, weights: np.ndarray, beat_period: float, beats_per_bar: int) -> float:
        bar_period = beat_period * beats_per_bar
        if bar_period <= 0:
            return 0.0

        phase_grid = np.linspace(0.0, beat_period, 32, endpoint=False)
        best = 0.0
        tol = max(0.02, 0.16 * beat_period)
        for phase in phase_grid:
            rel = (onsets - phase) / beat_period
            nearest_beat = np.rint(rel)
            dist = np.abs(rel - nearest_beat) * beat_period
            near = dist <= tol
            if not np.any(near):
                continue

            beat_idx = np.mod(nearest_beat.astype(np.int64), beats_per_bar)
            accents = np.zeros(beats_per_bar, dtype=np.float64)
            proximity = 1.0 - (dist[near] / tol)
            np.add.at(accents, beat_idx[near], weights[near] * proximity)
            total = float(np.sum(accents))
            if total <= 1e-9:
                continue

            downbeat = accents[0]
            others = float(np.mean(accents[1:])) if beats_per_bar > 1 else 0.0
            contrast = (downbeat - others) / (total + 1e-9)

            # Bar periodicity reinforcement.
            phase_in_bar = np.mod(onsets - phase, bar_period)
            bar_dist = np.minimum(phase_in_bar, bar_period - phase_in_bar)
            bar_fit = np.exp(-((bar_dist / tol) ** 2))
            periodicity = float(np.sum(bar_fit * weights) / (np.sum(weights) + 1e-12))

            score = 0.7 * contrast + 0.3 * periodicity
            if score > best:
                best = score
        return float(max(0.0, best))

    def _subdivision_alignment_score(
        self, onsets: np.ndarray, weights: np.ndarray, beat_period: float, subdivisions: int
    ) -> float:
        if subdivisions <= 0:
            return 0.0
        phase = np.mod(onsets, beat_period)
        sub_grid = np.linspace(0.0, beat_period, subdivisions, endpoint=False)
        dist = np.min(np.abs(phase[:, None] - sub_grid[None, :]), axis=1)
        dist = np.minimum(dist, beat_period - dist)
        tol = max(0.015, 0.1 * beat_period)
        fit = np.exp(-((dist / tol) ** 2))
        return float(np.sum(fit * weights) / (np.sum(weights) + 1e-12))

    def _signature_to_tactus_beats(self, ts: tuple[int, int]) -> tuple[int, bool]:
        num, den = ts
        if den == 8 and num in (6, 9, 12):
            return num // 3, True
        return num, False
