import React, { useState } from 'react';
import { ParentInputs } from '../types';
import { Sparkles, Send, Loader2, Bot, HelpCircle, Lightbulb, MessageSquare } from 'lucide-react';

interface AIAdvisorTabProps {
  inputs: ParentInputs;
}

export const AIAdvisorTab: React.FC<AIAdvisorTabProps> = ({ inputs }) => {
  const [prompt, setPrompt] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sampleQuestions = [
    'What if my child decides not to go to college? How does SECURE 2.0 protect my 529 investment?',
    'Can I open both a 529 Plan and a Trump Account ($5k limit) for the same child at the same time?',
    'How do I legally document earned income for a 10-year-old child to qualify for a Custodial Roth IRA?',
    'What are the state tax deduction limits for 529 contributions in my state?',
    'Which account is better if I want to help my child buy their first home at age 25?',
  ];

  const handleAskAI = async (queryText?: string) => {
    const textToSubmit = queryText || prompt;
    if (!textToSubmit.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch('/api/ai-advisor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: textToSubmit,
          childAge: inputs.childCurrentAge,
          monthlyContribution: inputs.monthlyContribution,
          investmentGoal: inputs.primaryGoal,
          state: inputs.state,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to get AI recommendation');
      }

      setResponse(data.text);
    } catch (err: any) {
      setError(err.message || 'An error occurred while connecting to the AI financial advisor.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-bold uppercase tracking-wider mb-2">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" /> AI Child Wealth Strategist (Gemini 2.5 Flash)
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Ask the AI Financial Advisor
          </h2>
          <p className="text-slate-500 dark:text-slate-400 text-xs sm:text-sm mt-1.5 leading-relaxed">
            Get instant, tailored advice on tax strategies, state-specific 529 deductions, SECURE 2.0 Roth rollovers, and child earned income compliance based on your family's scenario.
          </p>
        </div>
      </div>

      {/* Input Box */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-xs space-y-4">
        <label className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-indigo-600" /> What would you like to ask?
        </label>

        <div className="relative">
          <textarea
            id="ai-prompt-textarea"
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. Can I contribute to both a 529 and a Trump account? What happens at age 18?"
            className="w-full text-xs sm:text-sm p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-indigo-500"
          />
          <button
            id="submit-ai-prompt-button"
            onClick={() => handleAskAI()}
            disabled={loading || !prompt.trim()}
            className="absolute bottom-3 right-3 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs flex items-center gap-2 transition-colors disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            Ask Advisor
          </button>
        </div>

        {/* Sample Question Chips */}
        <div className="space-y-2 pt-2">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1">
            <Lightbulb className="w-3.5 h-3.5 text-amber-500" /> Common Questions:
          </span>
          <div className="flex flex-wrap gap-2">
            {sampleQuestions.map((q, idx) => (
              <button
                key={idx}
                id={`sample-question-${idx}`}
                onClick={() => {
                  setPrompt(q);
                  handleAskAI(q);
                }}
                className="text-xs text-left px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors"
              >
                "{q}"
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <Loader2 className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
          <div className="text-sm font-bold text-slate-900 dark:text-white">
            Analyzing your scenario with Gemini...
          </div>
          <p className="text-xs text-slate-500">
            Evaluating tax rules, 529 state limits, and rollover math for age {inputs.childCurrentAge}...
          </p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs">
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xs space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Bot className="w-5 h-5 text-emerald-600" />
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              AI Financial Advisor Response
            </h3>
          </div>

          <div className="prose dark:prose-invert max-w-none text-xs sm:text-sm text-slate-700 dark:text-slate-300 whitespace-pre-wrap leading-relaxed">
            {response}
          </div>
        </div>
      )}
    </div>
  );
};
