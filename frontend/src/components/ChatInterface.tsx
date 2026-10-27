'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Zap, Brain, Sparkles, Compass, CheckCircle2 } from 'lucide-react';

interface ChatInterfaceProps {
  sessionId: string;
  onSessionChange: (id: string) => void;
  useMatryoshka: boolean;
  onMatryoshkaChange: (val: boolean) => void;
  draftAtoms: number;
  onDraftAtomsChange: (val: number) => void;
  verifyAtoms: number;
  onVerifyAtomsChange: (val: number) => void;
  gammaSpec: number;
  onGammaSpecChange: (val: number) => void;
}

export function ChatInterface({
  sessionId,
  onSessionChange,
  useMatryoshka,
  onMatryoshkaChange,
  draftAtoms,
  onDraftAtomsChange,
  verifyAtoms,
  onVerifyAtomsChange,
  gammaSpec,
  onGammaSpecChange,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentToken, setCurrentToken] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentToken]);

  const starterPrompts = [
    {
      title: 'Needle-in-a-Haystack Passkey',
      desc: 'Test 100% retrieval protection at >2,000 tokens',
      prompt: 'System prompt: Maintain factual anchors.\n\nContext log:\nSystem report alpha: The secret passkey is 94812. All replica nodes are healthy and running on port 8443.\n\nQuestion: What is the secret passkey?',
    },
    {
      title: 'Matryoshka Speculative Test',
      desc: 'Verify draft m=4 against verify s=16',
      prompt: 'Write a comprehensive high-performance Python class implementing a multi-producer multi-consumer ring buffer with atomic sequence numbers.',
    },
    {
      title: 'HMM Context Topic Drift',
      desc: 'Observe β decay switch from 0.99 to 0.65',
      prompt: 'First discuss the CSS layout specifications for flexbox and grid alignment. Then abruptly transition to explaining distributed consensus algorithms in Raft.',
    },
  ];

  const handleSubmit = async (e?: React.FormEvent, directPrompt?: string) => {
    if (e) e.preventDefault();
    const promptToSend = directPrompt || input;
    if (!promptToSend.trim() || isGenerating) return;

    const userMessage = { role: 'user', content: promptToSend };
    setMessages(prev => [...prev, userMessage]);
    if (!directPrompt) setInput('');
    setIsGenerating(true);
    setCurrentToken('');
    let assembledResponse = '';

    const body = {
      prompt: promptToSend,
      session_id: sessionId || undefined,
      max_new_tokens: 128,
      temperature: 0.7,
      use_matryoshka: useMatryoshka,
      draft_atoms: draftAtoms,
      verify_atoms: verifyAtoms,
      gamma_spec: gammaSpec,
    };

    try {
      const response = await fetch('/api/v1/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status} - Generation failed`);

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'token') {
                  assembledResponse += data.text || '';
                  setCurrentToken(assembledResponse);
                } else if (data.type === 'done') {
                  setMessages(prev => [...prev, { role: 'assistant', content: assembledResponse }]);
                  setCurrentToken('');
                  setIsGenerating(false);
                  if (data.session_id) onSessionChange(data.session_id);
                } else if (data.type === 'start' && data.session_id) {
                  onSessionChange(data.session_id);
                } else if (data.type === 'error') {
                  setMessages(prev => [...prev, { role: 'assistant', content: `⚠️ Backend error: ${data.message}` }]);
                  setCurrentToken('');
                  setIsGenerating(false);
                }
              } catch {}
            }
          }
        }
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `⚠️ Request failed: ${String(err)}. Check that the backend is running.` },
      ]);
    } finally {
      setCurrentToken('');
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0 bg-bg">
      {/* Top Controls Toolbar */}
      <div className="px-5 py-3 border-b border-border bg-card/60 backdrop-blur flex flex-wrap items-center justify-between gap-3">
        {/* Speculative Decoding Toggle */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs font-semibold cursor-pointer bg-bg/80 hover:bg-bg border border-border px-3 py-1.5 rounded-lg transition select-none">
            <input
              type="checkbox"
              checked={useMatryoshka}
              onChange={(e) => onMatryoshkaChange(e.target.checked)}
              className="w-3.5 h-3.5 accent-primary rounded cursor-pointer"
            />
            <Zap className={`w-3.5 h-3.5 ${useMatryoshka ? 'text-amber-400 fill-amber-400' : 'text-gray-400'}`} />
            <span className={useMatryoshka ? 'text-white' : 'text-gray-300'}>Matryoshka Speculative</span>
          </label>

          {useMatryoshka && (
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className="text-gray-400 text-[11px]">Draft m:</span>
              <select
                value={draftAtoms}
                onChange={(e) => onDraftAtomsChange(Number(e.target.value))}
                className="bg-bg border border-border rounded px-2 py-1 text-xs text-sky-300 font-semibold focus:outline-none focus:border-primary"
              >
                <option value={2}>2 atoms</option>
                <option value={4}>4 atoms (optimal)</option>
                <option value={8}>8 atoms</option>
              </select>

              <span className="text-gray-400 text-[11px] ml-1">Verify s:</span>
              <select
                value={verifyAtoms}
                onChange={(e) => onVerifyAtomsChange(Number(e.target.value))}
                className="bg-bg border border-border rounded px-2 py-1 text-xs text-emerald-300 font-semibold focus:outline-none focus:border-primary"
              >
                <option value={8}>8 atoms</option>
                <option value={12}>12 atoms</option>
                <option value={16}>16 atoms (full)</option>
              </select>

              <span className="text-gray-400 text-[11px] ml-1">Lookahead γ:</span>
              <select
                value={gammaSpec}
                onChange={(e) => onGammaSpecChange(Number(e.target.value))}
                className="bg-bg border border-border rounded px-2 py-1 text-xs text-amber-300 font-semibold focus:outline-none focus:border-primary"
              >
                <option value={2}>2 tokens</option>
                <option value={4}>4 tokens</option>
                <option value={8}>8 tokens</option>
              </select>
            </div>
          )}
        </div>

        {/* Active Session Badge */}
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-400 bg-bg/60 px-2.5 py-1 rounded border border-border">
          <span>Session:</span>
          <span className="text-emerald-400 font-bold">{sessionId ? sessionId.slice(0, 10) : 'auto-generate'}</span>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-5 space-y-4" ref={messagesEndRef}>
        {messages.length === 0 ? (
          <div className="max-w-xl mx-auto my-auto pt-8 pb-4 text-center space-y-6">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-sky-500/20 border border-emerald-500/30 text-emerald-400 shadow-lg">
              <Brain className="w-8 h-8" />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-bold text-white tracking-tight">StoikovKV Compression Engine</h2>
              <p className="text-xs text-gray-300 max-w-md mx-auto leading-relaxed">
                Adaptive 3-tier KV caching: <span className="text-emerald-400 font-semibold">100% needle retrieval</span> in FP16,
                heavy-hitter eviction for peaked heads, and sparse dictionary coding for diffuse attention.
              </p>
            </div>

            {/* Starter Prompt Cards */}
            <div className="text-left space-y-2.5 pt-2">
              <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider px-1 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Quick Demonstration Prompts
              </div>
              <div className="grid grid-cols-1 gap-2">
                {starterPrompts.map((card, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSubmit(undefined, card.prompt)}
                    className="p-3 bg-card/60 hover:bg-card border border-border/80 hover:border-primary/50 rounded-xl transition text-left group flex items-start justify-between gap-3 shadow-sm"
                  >
                    <div>
                      <div className="text-xs font-semibold text-white group-hover:text-primary transition">
                        {card.title}
                      </div>
                      <div className="text-[11px] text-gray-400 mt-0.5">{card.desc}</div>
                    </div>
                    <Compass className="w-4 h-4 text-gray-500 group-hover:text-primary transition flex-shrink-0 mt-0.5" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[85%] px-4 py-3 rounded-2xl shadow-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-emerald-600 to-teal-700 text-white rounded-br-sm'
                    : 'bg-card border border-border text-slate-100 rounded-bl-sm'
                }`}
              >
                <div className="text-[10px] uppercase font-bold tracking-wider mb-1 opacity-70">
                  {msg.role === 'user' ? 'User' : 'StoikovKV Engine'}
                </div>
                <p className="text-sm whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{msg.content}</p>
              </div>
            </div>
          ))
        )}

        {/* Live Streaming Token Preview */}
        {isGenerating && (
          <div className="flex justify-start">
            <div className="bg-card border border-border text-slate-100 rounded-2xl rounded-bl-sm px-4 py-3 max-w-[85%] shadow-sm">
              <div className="text-[10px] uppercase font-bold tracking-wider text-emerald-400 mb-1 flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin" /> Streaming response...
              </div>
              <p className="text-sm whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
                {currentToken}
                <span className="inline-block w-2 h-4 bg-primary ml-1 animate-pulse align-middle" />
              </p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Box Footer */}
      <form onSubmit={(e) => handleSubmit(e)} className="p-4 border-t border-border bg-card/60 backdrop-blur">
        <div className="flex items-end gap-2 bg-bg border border-border/80 focus-within:border-primary/80 rounded-xl p-2 transition shadow-inner">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isGenerating ? 'Generating compressed inference...' : 'Type a prompt (Enter to send, Shift+Enter for newline)...'}
            disabled={isGenerating}
            className="flex-1 bg-transparent px-2 py-1.5 text-sm text-white placeholder-gray-500 resize-none min-h-[44px] max-h-[140px] focus:outline-none"
            rows={1}
          />
          <button
            type="submit"
            disabled={!input.trim() || isGenerating}
            className="bg-primary hover:bg-emerald-600 disabled:opacity-40 disabled:cursor-not-allowed text-white p-2.5 rounded-lg transition flex-shrink-0 shadow-sm"
            title="Send prompt"
          >
            {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </form>
    </div>
  );
}
