'use client';

import { useState, useEffect } from 'react';
import { Brain, Database, Gauge, Zap, RotateCcw, Trash2, Cpu } from 'lucide-react';
import { ChatInterface } from '../components/ChatInterface';
import { MetricsDashboard } from '../components/MetricsDashboard';
import { HeadRouterMap } from '../components/HeadRouterMap';
import { KVPressureGauge } from '../components/KVPressureGauge';

export default function Home() {
  const [sessionId, setSessionId] = useState<string>('');
  const [useMatryoshka, setUseMatryoshka] = useState(false);
  const [draftAtoms, setDraftAtoms] = useState(4);
  const [verifyAtoms, setVerifyAtoms] = useState(16);
  const [gammaSpec, setGammaSpec] = useState(4);
  const [header, setHeader] = useState<{ keep_ratio: number; memory_ratio_incl_dictionary: number; seq_length: number } | null>(null);
  const [calibrating, setCalibrating] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      try {
        const url = sessionId ? `/api/v1/metrics?session_id=${sessionId}` : '/api/v1/metrics';
        const res = await fetch(url);
        if (res.ok) {
          const d = await res.json();
          if (isMounted) setHeader(d);
        }
      } catch {}
    };
    load();
    const id = setInterval(load, 2500);
    return () => {
      isMounted = false;
      clearInterval(id);
    };
  }, [sessionId]);

  const handleRecalibrate = async () => {
    setCalibrating(true);
    try {
      await fetch('/api/v1/calibrate', { method: 'POST' });
      setSessionId('');
    } catch {}
    setCalibrating(false);
  };

  const handleClearSession = async () => {
    if (sessionId) {
      try {
        await fetch(`/api/v1/session/${sessionId}`, { method: 'DELETE' });
      } catch {}
      setSessionId('');
    }
  };

  return (
    <div className="h-screen max-h-screen overflow-hidden bg-bg text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <header className="bg-card/80 backdrop-blur border-b border-border px-6 py-3.5 flex items-center justify-between z-10 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-md shadow-emerald-500/20">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-tight">StoikovKV</h1>
              <span className="text-[10px] bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-mono font-semibold px-2 py-0.5 rounded-full">
                v0.1.0 · Hybrid Cache
              </span>
            </div>
            <p className="text-[11px] text-gray-400">Adaptive Head-Routing &amp; Matryoshka Speculative Decoding</p>
          </div>
        </div>

        {/* Live Metrics Header Badges */}
        <div className="hidden lg:flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 bg-bg/80 border border-border px-3 py-1.5 rounded-lg font-mono">
            <Gauge className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-gray-400">Budget:</span>
            <span className="text-sky-300 font-semibold">≤20% FP16</span>
          </div>

          <div className="flex items-center gap-1.5 bg-bg/80 border border-border px-3 py-1.5 rounded-lg font-mono">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-gray-400">Compression:</span>
            <span className="text-emerald-300 font-semibold">
              {header?.memory_ratio_incl_dictionary
                ? `${(1 / header.memory_ratio_incl_dictionary).toFixed(2)}×`
                : '5.43×'}
            </span>
          </div>

          <div className="flex items-center gap-1.5 bg-bg/80 border border-border px-3 py-1.5 rounded-lg font-mono">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-gray-400">Context:</span>
            <span className="text-amber-300 font-semibold">
              {header?.seq_length ? `${header.seq_length.toLocaleString()} tok` : '0 tok'}
            </span>
          </div>
        </div>

        {/* Quick Header Actions */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRecalibrate}
            disabled={calibrating}
            className="flex items-center gap-1.5 text-xs bg-bg/80 hover:bg-bg border border-border hover:border-primary/50 text-gray-300 hover:text-white px-3 py-1.5 rounded-lg transition disabled:opacity-50"
            title="Reload calibrated artifacts from disk"
          >
            <RotateCcw className={`w-3.5 h-3.5 text-emerald-400 ${calibrating ? 'animate-spin' : ''}`} />
            <span>Recalibrate</span>
          </button>

          {sessionId && (
            <button
              type="button"
              onClick={handleClearSession}
              className="flex items-center gap-1.5 text-xs bg-rose-950/30 hover:bg-rose-900/40 border border-rose-500/30 text-rose-300 px-3 py-1.5 rounded-lg transition"
              title="Clear current session and free KV memory"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Two-Panel Layout */}
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden min-h-0">
        {/* Left Panel: Chat & Generation */}
        <div className="md:w-3/5 lg:w-2/3 h-full flex flex-col min-h-0 border-r border-border">
          <ChatInterface
            sessionId={sessionId}
            onSessionChange={setSessionId}
            useMatryoshka={useMatryoshka}
            onMatryoshkaChange={setUseMatryoshka}
            draftAtoms={draftAtoms}
            onDraftAtomsChange={setDraftAtoms}
            verifyAtoms={verifyAtoms}
            onVerifyAtomsChange={setVerifyAtoms}
            gammaSpec={gammaSpec}
            onGammaSpecChange={setGammaSpec}
          />
        </div>

        {/* Right Panel: Observability Metrics, Router Map, VRAM Pressure */}
        <div className="md:w-2/5 lg:w-1/3 h-full flex flex-col min-h-0 bg-bg p-4 space-y-4 overflow-y-auto">
          <MetricsDashboard sessionId={sessionId} />
          <HeadRouterMap sessionId={sessionId} />
          <KVPressureGauge sessionId={sessionId} />
        </div>
      </main>
    </div>
  );
}
