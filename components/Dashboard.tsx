
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, AreaChart, Area } from 'recharts';
import { geminiService } from '../services/geminiService';

const data = [
  { name: 'Mon', leads: 400, conv: 240 },
  { name: 'Tue', leads: 300, conv: 139 },
  { name: 'Wed', leads: 200, conv: 980 },
  { name: 'Thu', leads: 278, conv: 390 },
  { name: 'Fri', leads: 189, conv: 480 },
  { name: 'Sat', leads: 239, conv: 380 },
  { name: 'Sun', leads: 349, conv: 430 },
];

const Dashboard: React.FC = () => {
  const [fastInsight, setFastInsight] = useState<string>('Loading quick insight...');

  useEffect(() => {
    geminiService.getQuickInsight('AI in SaaS sales').then(setFastInsight);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-slate-800">Sales Dashboard</h1>
        <div className="bg-indigo-50 border border-indigo-100 px-4 py-2 rounded-lg flex items-center gap-3 animate-pulse">
           <i className="fas fa-bolt text-indigo-600"></i>
           <span className="text-sm text-indigo-800 font-medium">Fast AI: {fastInsight}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Total Leads', value: '12,840', change: '+12%', icon: 'fa-users', color: 'bg-blue-500' },
          { label: 'Conversion Rate', value: '4.2%', change: '+0.5%', icon: 'fa-percentage', color: 'bg-green-500' },
          { label: 'Active Campaigns', value: '24', change: '-2', icon: 'fa-bullhorn', color: 'bg-purple-500' },
          { label: 'Avg. Lead Score', value: '78', change: '+5', icon: 'fa-star', color: 'bg-orange-500' },
        ].map((stat, i) => (
          <div key={i} className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-slate-500 text-sm font-medium">{stat.label}</p>
                <p className="text-2xl font-bold text-slate-800 mt-1">{stat.value}</p>
              </div>
              <div className={`${stat.color} w-10 h-10 rounded-lg flex items-center justify-center text-white`}>
                <i className={`fas ${stat.icon}`}></i>
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2">
              <span className={`text-xs font-semibold ${stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
                {stat.change}
              </span>
              <span className="text-xs text-slate-400">vs last month</span>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Lead Acquisition Trend</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorLeads" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <Tooltip />
                <Area type="monotone" dataKey="leads" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorLeads)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Campaign Performance</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                <Tooltip />
                <Bar dataKey="conv" fill="#818cf8" radius={[4, 4, 0, 0]} barSize={30} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
