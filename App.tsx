
import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import CampaignCreator from './components/CampaignCreator';
import SalesPitcher from './components/SalesPitcher';
import MarketAnalyzer from './components/MarketAnalyzer';
import LeadScorer from './components/LeadScorer';
import ChatBot from './components/ChatBot';
import { AppView } from './types';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<AppView>(AppView.DASHBOARD);

  const renderView = () => {
    switch (currentView) {
      case AppView.DASHBOARD:
        return <Dashboard />;
      case AppView.CAMPAIGNS:
        return <CampaignCreator />;
      case AppView.SALES_PITCH:
        return <SalesPitcher />;
      case AppView.MARKET_ANALYSIS:
        return <MarketAnalyzer />;
      case AppView.LEAD_SCORING:
        return <LeadScorer />;
      case AppView.INSIGHTS:
        return (
          <div className="flex flex-col items-center justify-center h-[60vh] text-center space-y-4">
             <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600">
               <i className="fas fa-microchip text-2xl"></i>
             </div>
             <h2 className="text-2xl font-bold text-slate-800">Advanced Business Insights</h2>
             <p className="text-slate-500 max-w-md">Our predictive analysis engine is currently processing your historical sales data. Predictive modeling will be available shortly.</p>
             <button onClick={() => setCurrentView(AppView.DASHBOARD)} className="text-indigo-600 font-bold hover:underline">Back to Dashboard</button>
          </div>
        );
      default:
        return <Dashboard />;
    }
  };

  const getTitle = () => {
    return currentView.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <Sidebar currentView={currentView} onViewChange={setCurrentView} />
      
      <main className="ml-64 min-h-screen p-8 lg:p-12">
        <header className="mb-10 flex justify-between items-center">
          <div>
            <h2 className="text-slate-400 text-sm font-bold uppercase tracking-widest mb-1">MarketMind Portal</h2>
            <h1 className="text-3xl font-black text-slate-900">{getTitle()}</h1>
          </div>
          <div className="flex items-center gap-6">
            <button className="w-10 h-10 rounded-full bg-white shadow-sm border border-slate-200 flex items-center justify-center text-slate-500 hover:text-indigo-600 transition-colors">
              <i className="fas fa-bell"></i>
            </button>
            <div className="flex items-center gap-3 pl-6 border-l border-slate-200">
              <div className="text-right">
                <p className="text-sm font-bold text-slate-900 leading-none">Alex Thompson</p>
                <p className="text-xs text-slate-400 mt-1">Growth Manager</p>
              </div>
              <img src="https://picsum.photos/40/40" alt="Profile" className="w-10 h-10 rounded-full border-2 border-white shadow-sm" />
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto">
          {renderView()}
        </div>
      </main>

      <ChatBot />
    </div>
  );
};

export default App;
