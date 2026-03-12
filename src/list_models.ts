import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({ 
  apiKey: process.env.GEMINI_API_KEY,
  apiVersion: 'v1beta'
});
async function run() {
  try {
    const response = await ai.models.list();
    for await (const model of response) {
      if (model.name.includes('embed')) {
        console.log(model.name);
      }
    }
  } catch (e: any) {
    console.error('Error:', e.message);
  }
}
run();
