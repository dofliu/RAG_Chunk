import type { EvaluationScores } from './types';

export interface ScorerWeights {
  faithfulness: number;
  relevance: number;
  correctness: number;
  latency: number;
}

const DEFAULT_WEIGHTS_WITH_GT: ScorerWeights = {
  faithfulness: 0.2,
  relevance: 0.2,
  correctness: 0.5,
  latency: 0.1,
};

const DEFAULT_WEIGHTS_NO_GT: ScorerWeights = {
  faithfulness: 0.4,
  relevance: 0.5,
  correctness: 0,
  latency: 0.1,
};

/**
 * Compute a composite score from 0-10.
 * - faithfulness, relevance, correctness are already 0-10
 * - latency is normalized: faster = higher score (max 10 at 0ms, 0 at >= 30s)
 * - hasExpectedAnswer: if true, correctness is weighted heavily; if false, ignored
 */
export function computeCompositeScore(
  scores: EvaluationScores,
  hasExpectedAnswer: boolean,
  weights?: ScorerWeights,
): number {
  const w = weights ?? (hasExpectedAnswer ? DEFAULT_WEIGHTS_WITH_GT : DEFAULT_WEIGHTS_NO_GT);
  const latencyScore = Math.max(0, 10 - (scores.latencyMs / 3000));
  const composite =
    w.faithfulness * scores.faithfulness +
    w.relevance * scores.relevance +
    w.correctness * scores.correctness +
    w.latency * latencyScore;
  return Math.round(composite * 100) / 100;
}
