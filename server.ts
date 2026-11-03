import express, { type Request, type Response } from 'express';
import { z } from 'zod';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

const AnalysisSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  summary: z.string(),
});

// Helper function to handle transient 503 errors with exponential backoff
async function generateWithRetry(contents: string, maxRetries = 3) {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      attempt++;
      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: contents,
        config: {
          systemInstruction: 'Analyze the text and return ONLY valid JSON matching: {"sentiment": "positive"|"negative"|"neutral", "summary": "string"}',
          responseMimeType: 'application/json',
        },
      });
      return response;
    } catch (error: any) {
      console.warn(`⚠️ Attempt ${attempt} failed: ${error.message}`);
      if (attempt >= maxRetries) throw error;

      // Exponential backoff delay (2s, 4s, 8s...)
      const delay = Math.pow(2, attempt) * 1000;
      console.log(`⏱️ Retrying in ${delay}ms due to high demand...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}

app.post('/api/analyze', async (req: Request, res: Response): Promise<void> => {
  try {
    const { text } = req.body;
    if (!text) {
      res.status(400).json({ error: 'Text field is required' });
      return;
    }

    // Call Gemini with built-in retry logic
    const response = await generateWithRetry(text);

    const rawText = response?.text || '{}';
    const parsedJson = JSON.parse(rawText);
    const validatedData = AnalysisSchema.parse(parsedJson);

    res.status(200).json({ success: true, data: validatedData });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Gemini backend server running on port ${PORT}`);
});