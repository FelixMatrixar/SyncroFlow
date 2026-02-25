import { GoogleGenAI } from '@google/genai';

let connectionSettings: any;

async function getCredentials() {
  

  // Fallback to environment variable
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('Gemini not connected and GEMINI_API_KEY not found');
  }
  return { apiKey };
}

async function getApiKey() {
  const { apiKey } = await getCredentials();
  return apiKey;
}

// WARNING: Never cache this client.
// Access tokens expire, so a new client must be created each time.
// Always call this function again to get a fresh client.
export async function getGeminiClient() {
  const apiKey = await getApiKey();
  return new GoogleGenAI({ apiKey });
}
