import { GoogleGenAI } from '@google/genai';
import type { EmbeddingProvider } from './types';

export const geminiProvider: EmbeddingProvider = {
  name: 'gemini',
  label: 'Google Gemini',
  models: [
    'gemini-embedding-2-preview',
    'gemini-embedding-001',
  ],
  requiresApiKey: true,

  async embedBatch(texts: string[], model: string, apiKey: string): Promise<number[][]> {
    const ai = new GoogleGenAI({ apiKey });
    const result = await ai.models.embedContent({
      model,
      contents: texts,
    });
    return result.embeddings.map((e: any) => e.values);
  },
};
