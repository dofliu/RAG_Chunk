import type { EmbeddingProvider, ProviderModel } from './types';
import { geminiProvider } from './gemini';
import { openaiProvider } from './openai';

const providers: EmbeddingProvider[] = [
  geminiProvider,
  openaiProvider,
];

export function getAllProviders(): EmbeddingProvider[] {
  return providers;
}

export function getProvider(name: string): EmbeddingProvider | undefined {
  return providers.find(p => p.name === name);
}

export function getAllModels(): ProviderModel[] {
  return providers.flatMap(p =>
    p.models.map(modelId => ({
      provider: p.name,
      modelId,
      label: `${modelId}`,
    }))
  );
}

export function getProviderForModel(modelId: string): EmbeddingProvider | undefined {
  return providers.find(p => p.models.includes(modelId));
}

/** Embed a batch of texts using the correct provider for the given model */
export async function embedWithModel(
  texts: string[],
  modelId: string,
  apiKeys: Record<string, string>,
): Promise<number[][]> {
  const provider = getProviderForModel(modelId);
  if (!provider) {
    throw new Error(`No provider found for model: ${modelId}`);
  }
  const apiKey = apiKeys[provider.name];
  if (provider.requiresApiKey && !apiKey) {
    throw new Error(`API key required for ${provider.label}. Please set it in the API Keys panel.`);
  }
  return provider.embedBatch(texts, modelId, apiKey);
}

// --- API Key persistence via localStorage ---
const API_KEYS_STORAGE_KEY = 'rag_lab_api_keys';

export function loadApiKeys(): Record<string, string> {
  try {
    const raw = localStorage.getItem(API_KEYS_STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

export function saveApiKeys(keys: Record<string, string>): void {
  localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(keys));
}
