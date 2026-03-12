import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  apiVersion: 'v1beta'
});

async function test() {
  const models = ['text-embedding-004', 'embedding-001', 'gemini-embedding-2-preview'];
  for (const model of models) {
    try {
      console.log(`Testing single embed for ${model}...`);
      const resSingle = await ai.models.embedContent({
        model: model,
        contents: 'test single'
      });
      console.log(`[SUCCESS SINGLE] ${model}:`, resSingle.embeddings?.[0]?.values?.length);
    } catch (e: any) {
      console.error(`[ERROR SINGLE] ${model}:`, e.message);
    }

    try {
      console.log(`Testing batch embed for ${model}...`);
      const resBatch = await ai.models.embedContent({
        model: model,
        contents: ['test batch 1', 'test batch 2']
      });
      console.log(`[SUCCESS BATCH] ${model}:`, resBatch.embeddings?.length);
    } catch (e: any) {
      console.error(`[ERROR BATCH] ${model}:`, e.message);
    }
  }
}

test();
