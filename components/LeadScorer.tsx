
import React, { useState } from 'react';
import { geminiService } from '../services/geminiService';
import { Lead } from '../types';

const LeadScorer: React.FC = () => {
  const [input, setInput] = useState('');
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState(false);

  const handleScore = async () => {
    if (!input) return;
    setLoading(true);
    try {
      const result = await geminiService.scoreLeads(input);
      setLeads(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const mockLeadsText = `John Doe, Tesla, high interest in CRM
Sarah Connor, Skynet, looking for security automation
Walter White, Blue Meth Co, basic inquiry`;

  return (
    <div className="space-y-8">
      <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
        <h2 className="text-xl font-bold text-slate-800 mb-4">Lead Intelligence Scoring</h2>
        <p className="text-slate-500 mb-6 text-sm">Paste lead information, LinkedIn bios, or email threads to score prospect potential.</p>
        <div className="space-y-4">
          <textarea
            rows={5}
            value={input}
            onChange={e => setInput(e.target.value)}
            className="w-full border border-slate-200 rounded-xl px-4 py-3 focus:ring-2 focus:ring-indigo-500/20 focus:outline-none text-sm"
            placeholder="e.g. Name, Company, Interactions..."
          />
          <div className="flex justify-between items-center">
            <button 
              onClick={() => setInput(mockLeadsText)}
              className="text-indigo-600 text-xs font-bold hover:underline"
            >
              Load Example Data
            </button>
            <button 
              onClick={handleScore}
              disabled={loading}
              className="bg-indigo-600 text-white font-bold px-8 py-2 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
              {loading ? 'Analyzing...' : 'Predict Scores'}
            </button>
          </div>
        </div>
      </div>

      {leads.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {leads.map((lead, i) => (
            <div key={i} className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 relative overflow-hidden group">
              <div className={`absolute top-0 right-0 w-1 h-full ${
                lead.score >= 80 ? 'bg-green-500' : lead.score >= 50 ? 'bg-orange-500' : 'bg-red-500'
              }`}></div>
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h4 className="font-bold text-slate-800">{lead.name}</h4>
                  <p className="text-xs text-slate-500 font-medium">{lead.company}</p>
                </div>
                <div className={`text-xl font-black ${
                  lead.score >= 80 ? 'text-green-600' : lead.score >= 50 ? 'text-orange-600' : 'text-red-600'
                }`}>
                  {lead.score}
                </div>
              </div>
              <p className="text-sm text-slate-600 line-clamp-3 mb-4">{lead.reasoning}</p>
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded ${
                  lead.status === 'hot' ? 'bg-red-100 text-red-700' : lead.status === 'warm' ? 'bg-orange-100 text-orange-700' : 'bg-blue-100 text-blue-700'
                }`}>
                  {lead.status}
                </span>
                <button className="text-xs text-indigo-600 font-bold ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                  View Detail →
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default LeadScorer;
