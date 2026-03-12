import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({ 
  apiKey: process.env.GEMINI_API_KEY
});
async function run() {
  try {
    const response = await ai.models.embedContent({
      model: 'gemini-embedding-2-preview',
      contents: 'hello'
    });
    console.log('gemini-embedding-2-preview singular success');
  } catch (e: any) {
    console.error('gemini-embedding-2-preview singular error:', e.message);
  }
  
  try {
    const response = await ai.models.embedContent({
      model: 'gemini-embedding-2-preview',
      contents: ['hello', 'world']
    });
    console.log('gemini-embedding-2-preview batch success');
  } catch (e: any) {
    console.error('gemini-embedding-2-preview batch error:', e.message);
  }
}
run();
