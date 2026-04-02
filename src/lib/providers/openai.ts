import type { EmbeddingProvider } from './types';

export const openaiProvider: EmbeddingProvider = {
  name: 'openai',
  label: 'OpenAI',
  models: [
    'text-embedding-3-small',
    'text-embedding-3-large',
  ],
  requiresApiKey: true,

  async embedBatch(texts: string[], model: string, apiKey: string): Promise<number[][]> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ input: texts, model }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(`OpenAI API error: ${err.error?.message || response.statusText}`);
    }

    const data = await response.json();
    // OpenAI returns embeddings sorted by index
    const sorted = data.data.sort((a: any, b: any) => a.index - b.index);
    return sorted.map((item: any) => item.embedding);
  },
};
