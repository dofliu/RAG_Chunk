import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  apiVersion: 'v1beta'
});

async function test() {
  const models = ['text-embedding-004', 'embedding-001', 'gemini-embedding-001', 'gemini-embedding-2-preview'];
  for (const model of models) {
    try {
      const res = await ai.models.embedContent({
        model: model,
        contents: 'test'
      });
      console.log(`[SUCCESS] ${model}:`, res.embeddings?.[0]?.values?.length);
    } catch (e: any) {
      console.error(`[ERROR] ${model}:`, e.message);
    }
  }
}

test();
