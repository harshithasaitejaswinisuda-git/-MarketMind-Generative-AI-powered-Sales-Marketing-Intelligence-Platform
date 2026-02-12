
import React, { useState } from 'react';
import { geminiService } from '../services/geminiService';
import { Campaign } from '../types';

const CampaignCreator: React.FC = () => {
  const [formData, setFormData] = useState({ name: '', audience: '', goals: '' });
  const [campaign, setCampaign] = useState<Campaign | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const result = await geminiService.generateCampaign(formData);
      setCampaign(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
        <h2 className="text-xl font-bold text-slate-800 mb-6">Create New Campaign</h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Campaign Name</label>
            <input
              required
              type="text"
              placeholder="e.g. Q4 Growth Sprint"
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
              value={formData.name}
              onChange={e => setFormData({...formData, name: e.target.value})}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Target Audience</label>
            <input
              required
              type="text"
              placeholder="e.g. CMOs of Mid-market SaaS companies"
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
              value={formData.audience}
              onChange={e => setFormData({...formData, audience: e.target.value})}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Campaign Goals</label>
            <textarea
              required
              rows={4}
              placeholder="What are you trying to achieve?"
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
              value={formData.goals}
              onChange={e => setFormData({...formData, goals: e.target.value})}
            ></textarea>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 text-white font-bold py-3 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-magic"></i>}
            {loading ? 'Generating Campaign...' : 'Generate with Gemini'}
          </button>
        </form>
      </div>

      <div className="bg-slate-50 border-2 border-dashed border-slate-200 rounded-2xl p-8 flex flex-col">
        {campaign ? (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 animate-in fade-in slide-in-from-right-4 duration-500">
            <div className="flex justify-between items-center mb-6">
              <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded">GEN AI OUTPUT</span>
              <button className="text-indigo-600 text-sm font-bold hover:underline">Download PDF</button>
            </div>
            <div className="prose prose-slate max-w-none whitespace-pre-wrap">
              {campaign.content}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-400 text-center">
            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
              <i className="fas fa-file-alt text-2xl"></i>
            </div>
            <p className="font-medium">Your AI-generated campaign content will appear here.</p>
            <p className="text-sm">Fill out the form to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CampaignCreator;
