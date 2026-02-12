
import React, { useState } from 'react';
import { geminiService } from '../services/geminiService';
import { MarketInsight } from '../types';

const MarketAnalyzer: React.FC = () => {
  const [query, setQuery] = useState('');
  const [analysis, setAnalysis] = useState<MarketInsight | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const result = await geminiService.analyzeMarket(query);
      setAnalysis(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
        <div className="flex gap-4">
          <input 
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="flex-1 border border-slate-200 rounded-xl px-6 py-4 text-lg focus:ring-2 focus:ring-indigo-500/20 focus:outline-none"
            placeholder="Enter industry or trend (e.g. 'Gen AI impact on real estate marketing 2025')"
          />
          <button 
            onClick={handleAnalyze}
            disabled={loading}
            className="bg-slate-900 text-white font-bold px-8 rounded-xl hover:bg-slate-800 transition-colors disabled:opacity-50 flex items-center gap-3"
          >
            {loading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-search"></i>}
            Analyze
          </button>
        </div>
      </div>

      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
            <h3 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-3">
              <i className="fas fa-file-contract text-indigo-600"></i>
              Deep Market Analysis
            </h3>
            <div className="prose prose-slate max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap">
              {analysis.summary}
            </div>
          </div>
          <div className="space-y-6">
            <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-100">
              <h4 className="font-bold text-indigo-900 mb-4 flex items-center gap-2">
                <i className="fas fa-link"></i>
                Grounding Sources
              </h4>
              <ul className="space-y-3">
                {analysis.sources.map((src, i) => (
                  <li key={i}>
                    <a 
                      href={src.uri} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center gap-2 group"
                    >
                      <i className="fas fa-external-link-alt text-[10px] opacity-0 group-hover:opacity-100 transition-opacity"></i>
                      <span className="truncate">{src.title}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-slate-900 p-6 rounded-2xl text-white">
              <h4 className="font-bold mb-2">Pro Tip</h4>
              <p className="text-slate-400 text-sm">
                This analysis uses real-time Google Search grounding. Use this data for strategic quarterly planning.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MarketAnalyzer;
