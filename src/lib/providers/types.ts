export interface EmbeddingProvider {
  /** Provider identifier, e.g. "gemini", "openai" */
  name: string;
  /** Human-readable label for the UI */
  label: string;
  /** List of model IDs this provider supports */
  models: string[];
  /** Whether this provider needs an API key */
  requiresApiKey: boolean;
  /** Embed a batch of texts, returns an array of embedding vectors */
  embedBatch(texts: string[], model: string, apiKey: string): Promise<number[][]>;
}

export interface ProviderModel {
  provider: string;
  modelId: string;
  label: string;
}
