import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Server-side Gemini AI Endpoint
  app.post('/api/ai-advisor', async (req, res) => {
    try {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        return res.status(500).json({
          error: 'GEMINI_API_KEY environment variable is missing on the server.'
        });
      }

      const { prompt, childAge, monthlyContribution, investmentGoal, state } = req.body;

      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });

      const systemInstruction = `You are a certified financial planner specializing in child investment accounts, education funding, tax strategy, and early wealth building in the United States.
You provide clear, objective, accurate, actionable advice covering:
1. 529 College Savings Plans (tax-free growth for education, SECURE 2.0 Act rollover up to $35k to Roth IRA after 15 years, state tax deductions).
2. Trump Accounts / Child Savings Accounts / Custodial IRA accounts ($5,000 yearly contribution limit, growth up to age 18, rollover strategies to Traditional or Roth IRA up to age 60/65).
3. UTMA / UGMA Custodial Accounts (flexibility, Kiddie tax thresholds, control at age 18/21).
4. Custodial Roth IRAs (requires child earned income, 100% tax-free growth, withdrawal rules).
5. Coverdell ESAs ($2,000/yr limit, education focus).
6. Taxable Brokerage / Trust accounts.

Keep your tone warm, professional, encouraging, and structured. Use formatting like bullet points and concise bold titles. Do not give binding legal tax advice, but explain financial rules clearly.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: `Parent Scenario Details:
- Child Current Age: ${childAge ?? '0'} years old
- Monthly Contribution Budget: $${monthlyContribution ?? 300}/month ($${(monthlyContribution ?? 300) * 12}/year)
- Primary Goal: ${investmentGoal ?? 'Balanced Education & General Wealth'}
- State of Residence: ${state ?? 'Not specified'}

User Question: ${prompt}`,
        config: {
          systemInstruction,
          temperature: 0.7,
        },
      });

      res.json({ text: response.text });
    } catch (err: any) {
      console.error('Error in /api/ai-advisor:', err);
      res.status(500).json({ error: err.message || 'Failed to generate AI advice' });
    }
  });

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  // Vite middleware setup
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Child Financial Planner Server running on http://localhost:${PORT}`);
  });
}

startServer();
