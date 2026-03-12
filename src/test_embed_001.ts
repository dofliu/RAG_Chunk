import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({ 
  apiKey: process.env.GEMINI_API_KEY,
  apiVersion: 'v1beta'
});
async function run() {
  try {
    const response = await ai.models.embedContent({
      model: 'embedding-001',
      contents: 'hello'
    });
    console.log('embedding-001 success');
  } catch (e: any) {
    console.error('embedding-001 error:', e.message);
  }
}
run();
