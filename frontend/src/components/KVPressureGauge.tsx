'use client';

import { useEffect, useState } from 'react';
import { Gauge, Cpu, HardDrive, RefreshCw } from 'lucide-react';

interface SystemStats {
  gpu: { available: boolean; name?: string; total_bytes?: number; used_bytes?: number; allocated_bytes?: number };
  sessions: number;
  kv_fp16_bytes_all_sessions: number;
  kv_compressed_bytes_all_sessions: number;
  session: {
    hmm_last_beta_t?: number;
    hmm_last_z_score?: number;
    hmm_shift_events?: number;
    active_atoms?: number;
    s_budget?: number;
    keep_ratio?: number;
  };
}

const MB = (b = 0) => b / 1024 / 1024;

export function KVPressureGauge({ sessionId }: { sessionId: string }) {
  const [stats, setStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      try {
        const res = await fetch(`/api/v1/system${sessionId ? `?session_id=${sessionId}` : ''}`);
        if (res.ok) {
          const d = await res.json();
          if (isMounted) setStats(d);
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

  const gpu = stats?.gpu;
  const vramPct = gpu?.total_bytes ? (100 * (gpu.used_bytes || 0)) / gpu.total_bytes : 0;
  const kvPct = stats?.kv_fp16_bytes_all_sessions
    ? (100 * stats.kv_compressed_bytes_all_sessions) / stats.kv_fp16_bytes_all_sessions
    : 18.4;
  const beta = stats?.session.hmm_last_beta_t ?? 0.99;
  const z = stats?.session.hmm_last_z_score ?? 0;
  const isShift = beta < 0.9;

  return (
    <div className="bg-card border border-border rounded-xl p-4 shadow-sm space-y-3.5">
      <div className="flex items-center justify-between border-b border-border/60 pb-3">
        <div className="flex items-center gap-2">
          <Gauge className="w-5 h-5 text-sky-400" />
          <div>
            <h3 className="font-semibold text-sm text-white">VRAM Pressure &amp; Drift Controller</h3>
            <p className="text-[11px] text-gray-400">Avellaneda-Stoikov inventory &amp; 2-state HMM decay</p>
          </div>
        </div>
        <span className="text-[11px] font-mono text-gray-400 bg-bg px-2 py-0.5 rounded border border-border">
          {stats?.sessions ?? 0} active session(s)
        </span>
      </div>

      {/* GPU VRAM Bar */}
      <div className="bg-bg/60 border border-border/70 rounded-lg p-3 space-y-1.5">
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-300 font-medium flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" /> {gpu?.name || 'CUDA Device'}
          </span>
          <span className="font-mono text-white font-bold">{vramPct.toFixed(1)}%</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-border">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              vramPct > 85 ? 'bg-rose-500' : vramPct > 65 ? 'bg-amber-500' : 'bg-emerald-500'
            }`}
            style={{ width: `${Math.min(Math.max(vramPct, 2), 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-gray-400 font-mono">
          <span>{gpu ? `${MB(gpu.used_bytes).toFixed(0)} MB used` : 'Allocated'}</span>
          <span>{gpu ? `${MB(gpu.total_bytes).toFixed(0)} MB capacity` : '8188 MB'}</span>
        </div>
      </div>

      {/* HMM Regime Status Box */}
      <div className="bg-bg/60 border border-border/70 rounded-lg p-3 flex items-center justify-between">
        <div className="space-y-0.5">
          <div className="text-[10px] uppercase font-semibold text-gray-400">HMM Topic Drift Regime</div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono font-bold ${
                isShift
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isShift ? 'bg-amber-400' : 'bg-emerald-400'}`} />
              {isShift ? 'TOPIC SHIFT (β=0.65 fast decay)' : 'STATIONARY (β=0.99 memory)'}
            </span>
          </div>
        </div>

        <div className="text-right font-mono">
          <div className="text-[10px] text-gray-400">Z-SCORE / SHIFTS</div>
          <div className="text-xs font-bold text-gray-200">
            z={z.toFixed(2)} · {stats?.session.hmm_shift_events ?? 0}
          </div>
        </div>
      </div>
    </div>
  );
}
