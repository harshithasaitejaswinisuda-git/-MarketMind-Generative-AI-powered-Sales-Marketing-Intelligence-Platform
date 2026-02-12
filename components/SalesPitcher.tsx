
import React, { useState } from 'react';
import { geminiService } from '../services/geminiService';
import { SalesPitch } from '../types';

const SalesPitcher: React.FC = () => {
  const [product, setProduct] = useState('');
  const [persona, setPersona] = useState('');
  const [pitch, setPitch] = useState<SalesPitch | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!product || !persona) return;
    setLoading(true);
    try {
      const result = await geminiService.generateSalesPitch(persona, product);
      setPitch(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="bg-indigo-900 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">Sales Pitch Architect</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-indigo-200 text-sm">Product or Service</label>
            <input 
              value={product}
              onChange={e => setProduct(e.target.value)}
              className="w-full bg-indigo-800/50 border border-indigo-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-400 focus:outline-none"
              placeholder="e.g. AI-driven CRM automation"
            />
          </div>
          <div className="space-y-2">
            <label className="text-indigo-200 text-sm">Target Buyer Persona</label>
            <input 
              value={persona}
              onChange={e => setPersona(e.target.value)}
              className="w-full bg-indigo-800/50 border border-indigo-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-400 focus:outline-none"
              placeholder="e.g. Overwhelmed Sales Manager"
            />
          </div>
        </div>
        <button 
          onClick={handleGenerate}
          disabled={loading}
          className="mt-8 bg-white text-indigo-900 font-bold px-8 py-3 rounded-lg hover:bg-indigo-50 transition-colors disabled:opacity-50"
        >
          {loading ? 'Crafting...' : 'Craft Perfect Pitch'}
        </button>
      </div>

      {pitch && (
        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
          <div className="border-b border-slate-100 p-6 flex justify-between items-center bg-slate-50/50">
            <h3 className="font-bold text-slate-800">Generated Pitch: {pitch.title}</h3>
            <div className="flex gap-2">
              <button className="p-2 hover:bg-slate-200 rounded-lg transition-colors"><i className="fas fa-copy text-slate-500"></i></button>
              <button className="p-2 hover:bg-slate-200 rounded-lg transition-colors"><i className="fas fa-share-alt text-slate-500"></i></button>
            </div>
          </div>
          <div className="p-8 prose prose-indigo max-w-none whitespace-pre-wrap leading-relaxed text-slate-700">
            {pitch.pitch}
          </div>
        </div>
      )}
    </div>
  );
};

export default SalesPitcher;
