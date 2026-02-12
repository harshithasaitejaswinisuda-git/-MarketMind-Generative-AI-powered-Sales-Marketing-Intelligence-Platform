
export enum AppView {
  DASHBOARD = 'dashboard',
  CAMPAIGNS = 'campaigns',
  SALES_PITCH = 'sales_pitch',
  MARKET_ANALYSIS = 'market_analysis',
  LEAD_SCORING = 'lead_scoring',
  INSIGHTS = 'insights'
}

export interface Campaign {
  id: string;
  name: string;
  targetAudience: string;
  channel: string;
  content: string;
  status: 'draft' | 'active' | 'completed';
}

export interface SalesPitch {
  id: string;
  title: string;
  persona: string;
  pitch: string;
  createdAt: string;
}

export interface Lead {
  id: string;
  name: string;
  company: string;
  status: 'cold' | 'warm' | 'hot';
  score: number;
  reasoning: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
}

export interface MarketInsight {
  topic: string;
  summary: string;
  sources: { title: string; uri: string }[];
}
