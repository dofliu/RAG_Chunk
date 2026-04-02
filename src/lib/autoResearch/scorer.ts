import type { EvaluationScores } from './types';

export interface ScorerWeights {
  faithfulness: number;
  relevance: number;
  latency: number;
}

const DEFAULT_WEIGHTS: ScorerWeights = {
  faithfulness: 0.4,
  relevance: 0.5,
  latency: 0.1,
};

/**
 * Compute a composite score from 0-10.
 * - faithfulness and relevance are already 0-10
 * - latency is normalized: faster = higher score (max 10 at 0ms, 0 at >= 30s)
 */
export function computeCompositeScore(
  scores: EvaluationScores,
  weights: ScorerWeights = DEFAULT_WEIGHTS,
): number {
  const latencyScore = Math.max(0, 10 - (scores.latencyMs / 3000));
  const composite =
    weights.faithfulness * scores.faithfulness +
    weights.relevance * scores.relevance +
    weights.latency * latencyScore;
  return Math.round(composite * 100) / 100;
}
