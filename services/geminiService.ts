
import { GoogleGenAI, Type } from "@google/genai";
import { Campaign, SalesPitch, MarketInsight, Lead } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

export const geminiService = {
  // Deep Thinking Chatbot - gemini-3-pro-preview
  async getChatResponse(history: { role: string; text: string }[], message: string) {
    const chat = ai.chats.create({
      model: 'gemini-3-pro-preview',
      config: {
        systemInstruction: 'You are a world-class sales and marketing consultant. Provide strategic, data-driven advice.',
        thinkingConfig: { thinkingBudget: 2000 }
      }
    });
    const result = await chat.sendMessage({ message });
    return result.text;
  },

  // Fast Insights - gemini-2.5-flash-lite-latest
  async getQuickInsight(topic: string): Promise<string> {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-lite-latest',
      contents: `Provide a 2-sentence marketing insight about ${topic}. Be extremely concise and fast.`,
    });
    return response.text || "No insight available.";
  },

  // Campaign Generation
  async generateCampaign(params: { name: string; audience: string; goals: string }): Promise<Campaign> {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Generate a marketing campaign for ${params.name}. 
      Target: ${params.audience}. 
      Goals: ${params.goals}. 
      Format as a professional structured plan including headline, body, and channels.`,
    });

    return {
      id: Math.random().toString(36).substr(2, 9),
      name: params.name,
      targetAudience: params.audience,
      channel: 'Multi-Channel',
      content: response.text || '',
      status: 'draft'
    };
  },

  // Sales Pitch Creation
  async generateSalesPitch(persona: string, product: string): Promise<SalesPitch> {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: `Create a high-converting sales pitch for ${product} targeting ${persona}. 
      Focus on pain points and specific value propositions.`,
    });

    return {
      id: Math.random().toString(36).substr(2, 9),
      title: `${product} for ${persona}`,
      persona,
      pitch: response.text || '',
      createdAt: new Date().toISOString(),
    };
  },

  // Market Analysis with Search Grounding
  async analyzeMarket(topic: string): Promise<MarketInsight> {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Analyze current market trends for: ${topic}. Focus on 2024-2025 data.`,
      config: {
        tools: [{ googleSearch: {} }],
      },
    });

    const sources = response.candidates?.[0]?.groundingMetadata?.groundingChunks
      ?.map((chunk: any) => ({
        title: chunk.web?.title || 'Source',
        uri: chunk.web?.uri || '#'
      }))
      .filter((s: any) => s.uri !== '#') || [];

    return {
      topic,
      summary: response.text || '',
      sources: sources.slice(0, 5)
    };
  },

  // Lead Scoring with structured JSON
  async scoreLeads(leadData: string): Promise<Lead[]> {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Score these leads based on potential value: ${leadData}`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              name: { type: Type.STRING },
              company: { type: Type.STRING },
              status: { type: Type.STRING },
              score: { type: Type.NUMBER },
              reasoning: { type: Type.STRING },
            },
            required: ["name", "company", "status", "score", "reasoning"],
          }
        }
      }
    });

    return JSON.parse(response.text || '[]');
  }
};
