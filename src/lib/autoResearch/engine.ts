import { GoogleGenAI } from '@google/genai';
import { chunkText, searchChunks, cosineSimilarity } from '../rag';
import { embedWithModel } from '../providers/registry';
import { computeCompositeScore } from './scorer';
import type {
  TestQuestion,
  SearchSpace,
  ExperimentConfig,
  ExperimentResult,
  AutoResearchProgress,
  AutoResearchReport,
} from './types';

/** Generate all experiment combinations from the search space */
function generateConfigs(space: SearchSpace): ExperimentConfig[] {
  const configs: ExperimentConfig[] = [];
  for (const embeddingModel of space.embeddingModels) {
    for (const chunkingStrategy of space.chunkingStrategies) {
      const sizes = chunkingStrategy === 'paragraph' ? [0] : space.chunkSizes;
      const overlaps = chunkingStrategy !== 'fixed' ? [0] : space.overlaps;
      for (const chunkSize of sizes) {
        for (const overlap of overlaps) {
          for (const kValue of space.kValues) {
            for (const llmModel of space.llmModels) {
              configs.push({ embeddingModel, chunkSize, overlap, chunkingStrategy, kValue, llmModel });
            }
          }
        }
      }
    }
  }
  return configs;
}

/** Chunk the document text based on strategy */
async function chunkByStrategy(
  text: string,
  strategy: 'fixed' | 'paragraph' | 'semantic',
  chunkSize: number,
  overlap: number,
  apiKeys: Record<string, string>,
): Promise<string[]> {
  if (strategy === 'paragraph') {
    return text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  }
  if (strategy === 'semantic') {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    if (sentences.length <= 1) return [text];

    // Use gemini for semantic chunking sentence embeddings
    const geminiKey = apiKeys['gemini'] || '';
    const sentenceEmbeddings: number[][] = [];
    const batchSize = 20;
    for (let i = 0; i < sentences.length; i += batchSize) {
      const batch = sentences.slice(i, i + batchSize);
      const vecs = await embedWithModel(batch, 'gemini-embedding-001', { gemini: geminiKey });
      sentenceEmbeddings.push(...vecs);
    }

    const chunks: string[] = [];
    let current = sentences[0];
    for (let i = 0; i < sentenceEmbeddings.length - 1; i++) {
      const sim = cosineSimilarity(sentenceEmbeddings[i], sentenceEmbeddings[i + 1]);
      if (sim > 0.7 && current.length < chunkSize) {
        current += ' ' + sentences[i + 1];
      } else {
        chunks.push(current);
        current = sentences[i + 1];
      }
    }
    chunks.push(current);
    return chunks;
  }
  // Fixed size
  return chunkText(text, chunkSize, overlap);
}

export interface RunAutoResearchOptions {
  documentText: string;
  testQuestions: TestQuestion[];
  searchSpace: SearchSpace;
  apiKeys: Record<string, string>;
  maxExperiments?: number;
  onProgress: (progress: AutoResearchProgress) => void;
  abortSignal?: AbortSignal;
}

export async function runAutoResearch(opts: RunAutoResearchOptions): Promise<AutoResearchReport> {
  const { documentText, testQuestions, searchSpace, apiKeys, onProgress, abortSignal } = opts;

  let allConfigs = generateConfigs(searchSpace);
  if (opts.maxExperiments && allConfigs.length > opts.maxExperiments) {
    // Random sample if too many combinations
    allConfigs = allConfigs.sort(() => Math.random() - 0.5).slice(0, opts.maxExperiments);
  }

  const total = allConfigs.length;
  const results: ExperimentResult[] = [];
  const startTime = Date.now();

  // Cache embeddings per (model, strategy, chunkSize, overlap) to avoid redundant API calls
  const embeddingCache: Record<string, { text: string; embedding: number[] }[]> = {};

  for (let i = 0; i < allConfigs.length; i++) {
    if (abortSignal?.aborted) break;

    const config = allConfigs[i];
    onProgress({
      current: i + 1,
      total,
      currentConfig: config,
      phase: 'chunking',
      message: `[${i + 1}/${total}] Chunking (${config.chunkingStrategy}, size=${config.chunkSize})...`,
    });

    try {
      // 1. Chunk
      const cacheKey = `${config.embeddingModel}_${config.chunkingStrategy}_${config.chunkSize}_${config.overlap}`;
      let chunkEmbeddings = embeddingCache[cacheKey];

      if (!chunkEmbeddings) {
        const chunks = await chunkByStrategy(
          documentText, config.chunkingStrategy, config.chunkSize, config.overlap, apiKeys,
        );

        // 2. Embed chunks
        onProgress({
          current: i + 1, total, currentConfig: config, phase: 'embedding',
          message: `[${i + 1}/${total}] Embedding ${chunks.length} chunks with ${config.embeddingModel}...`,
        });

        const embeddings: number[][] = [];
        const batchSize = 10;
        for (let b = 0; b < chunks.length; b += batchSize) {
          if (abortSignal?.aborted) break;
          const batch = chunks.slice(b, b + batchSize);
          const vecs = await embedWithModel(batch, config.embeddingModel, apiKeys);
          embeddings.push(...vecs);
        }

        chunkEmbeddings = chunks.map((text, idx) => ({ text, embedding: embeddings[idx] }));
        embeddingCache[cacheKey] = chunkEmbeddings;
      }

      // Run each test question and average the scores
      let totalFaithfulness = 0;
      let totalRelevance = 0;
      let totalLatency = 0;
      let lastAnswer = '';

      for (const tq of testQuestions) {
        if (abortSignal?.aborted) break;

        // 3. Embed query & search
        onProgress({
          current: i + 1, total, currentConfig: config, phase: 'searching',
          message: `[${i + 1}/${total}] Searching for: "${tq.question.substring(0, 40)}..."`,
        });

        const qVecs = await embedWithModel([tq.question], config.embeddingModel, apiKeys);
        const retrieved = searchChunks(qVecs[0], chunkEmbeddings, config.kValue);
        const context = retrieved.map((c, idx) => `[Chunk ${idx + 1}]:\n${c.text}`).join('\n\n---\n\n');

        // 4. Generate answer
        onProgress({
          current: i + 1, total, currentConfig: config, phase: 'generating',
          message: `[${i + 1}/${total}] Generating answer with ${config.llmModel}...`,
        });

        const genStart = Date.now();
        // Determine the API key for LLM generation - use gemini key for gemini models
        const llmApiKey = apiKeys['gemini'] || '';
        const genAi = new GoogleGenAI({ apiKey: llmApiKey });
        const answerResult = await genAi.models.generateContent({
          model: config.llmModel,
          contents: `You are a helpful assistant. Answer based ONLY on the provided context.\n\nContext:\n${context}\n\nQuestion: ${tq.question}`,
        });
        const answerText = answerResult.text || '';
        const latencyMs = Date.now() - genStart;
        lastAnswer = answerText;

        // 5. Evaluate
        onProgress({
          current: i + 1, total, currentConfig: config, phase: 'evaluating',
          message: `[${i + 1}/${total}] Evaluating answer quality...`,
        });

        const evalPrompt = `Evaluate the following RAG answer.
Metrics:
1. Faithfulness (0-10): Is the answer grounded ONLY in the context?
2. Relevance (0-10): Does the answer directly address the query?

Return ONLY a JSON object: { "faithfulness": number, "relevance": number }

Query: ${tq.question}
Context: ${context.substring(0, 2000)}
Answer: ${answerText}`;

        let faithfulness = 5;
        let relevance = 5;
        try {
          const evalResult = await genAi.models.generateContent({
            model: 'gemini-3-flash-preview',
            contents: evalPrompt,
            config: { responseMimeType: 'application/json' },
          });
          const evalData = JSON.parse(evalResult.text || '{}');
          faithfulness = evalData.faithfulness ?? 5;
          relevance = evalData.relevance ?? 5;
        } catch {
          // Keep defaults
        }

        totalFaithfulness += faithfulness;
        totalRelevance += relevance;
        totalLatency += latencyMs;
      }

      const numQ = testQuestions.length || 1;
      const scores = {
        faithfulness: totalFaithfulness / numQ,
        relevance: totalRelevance / numQ,
        latencyMs: totalLatency / numQ,
      };

      results.push({
        config,
        scores,
        compositeScore: computeCompositeScore(scores),
        answer: lastAnswer,
      });
    } catch (error: any) {
      // Record failed experiment with zero scores
      results.push({
        config,
        scores: { faithfulness: 0, relevance: 0, latencyMs: 0 },
        compositeScore: 0,
        answer: `Error: ${error.message}`,
      });
    }
  }

  // Sort by composite score descending
  results.sort((a, b) => b.compositeScore - a.compositeScore);

  onProgress({
    current: total, total, currentConfig: null, phase: 'done',
    message: `Completed ${results.length} experiments!`,
  });

  return {
    bestConfig: results[0],
    rankings: results,
    totalExperiments: results.length,
    totalTimeMs: Date.now() - startTime,
  };
}
