export interface TestQuestion {
  question: string;
  expectedAnswer?: string;
}

export interface SearchSpace {
  embeddingModels: string[];
  chunkSizes: number[];
  overlaps: number[];
  chunkingStrategies: ('fixed' | 'paragraph' | 'semantic')[];
  kValues: number[];
  llmModels: string[];
}

export interface ExperimentConfig {
  embeddingModel: string;
  chunkSize: number;
  overlap: number;
  chunkingStrategy: 'fixed' | 'paragraph' | 'semantic';
  kValue: number;
  llmModel: string;
}

export interface EvaluationScores {
  faithfulness: number;
  relevance: number;
  correctness: number;
  latencyMs: number;
}

export interface ExperimentResult {
  config: ExperimentConfig;
  scores: EvaluationScores;
  compositeScore: number;
  answer: string;
  reasoning?: string;
}

export interface AutoResearchProgress {
  current: number;
  total: number;
  currentConfig: ExperimentConfig | null;
  phase: 'idle' | 'chunking' | 'embedding' | 'searching' | 'generating' | 'evaluating' | 'done' | 'error';
  message: string;
}

export interface AutoResearchReport {
  bestConfig: ExperimentResult;
  rankings: ExperimentResult[];
  totalExperiments: number;
  totalTimeMs: number;
}
