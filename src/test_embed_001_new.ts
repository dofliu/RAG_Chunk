import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({ 
  apiKey: process.env.GEMINI_API_KEY
});
async function run() {
  try {
    const response = await ai.models.embedContent({
      model: 'gemini-embedding-001',
      contents: 'hello'
    });
    console.log('gemini-embedding-001 success');
  } catch (e: any) {
    console.error('gemini-embedding-001 error:', e.message);
  }
}
run();
