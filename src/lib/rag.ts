export function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  if (chunkSize <= 0) return [];
  if (overlap >= chunkSize) overlap = chunkSize - 1;

  const chunks: string[] = [];
  let i = 0;
  while (i < text.length) {
    chunks.push(text.slice(i, i + chunkSize));
    i += chunkSize - overlap;
  }
  return chunks;
}

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export interface ChunkWithScore {
  text: string;
  score: number;
}

export function searchChunks(queryEmbedding: number[], chunkEmbeddings: {text: string, embedding: number[]}[], topK: number = 3): ChunkWithScore[] {
  const scores = chunkEmbeddings.map(chunk => ({
    text: chunk.text,
    score: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));
  
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK);
}
