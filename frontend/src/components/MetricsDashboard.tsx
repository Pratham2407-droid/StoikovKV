'use client';

import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Database, Zap, Cpu, HardDrive } from 'lucide-react';

interface Telemetry {
  fp16_bytes?: number;
  compressed_bytes?: number;
  memory_ratio_vs_fp16?: number;
  memory_ratio_incl_dictionary?: number;
  bytes_read_last_step?: number;
  cumulative_bytes_read?: number;
  hmm_last_beta_t?: number;
  hmm_last_z_score?: number;
  hmm_shift_events?: number;
  seq_length?: number;
  active_atoms?: number;
  s_budget?: number;
  keep_ratio?: number;
  per_tier_token_counts?: {
    local_window_fp16?: number;
    retrieval_fp16?: number;
    peaked_h2o_fp16?: number;
    diffuse_lexico_sparse?: number;
    evicted_h2o?: number;
  };
}

interface MetricsDashboardProps {
  sessionId: string;
}

export function MetricsDashboard({ sessionId }: MetricsDashboardProps) {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = async () => {
      try {
        const url = sessionId ? `/api/v1/metrics?session_id=${sessionId}` : '/api/v1/metrics';
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          if (isMounted) setTelemetry(data);
        }
      } catch {}
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [sessionId]);

  const memPct = (telemetry?.memory_ratio_vs_fp16 ?? 0.181) * 100;
  const dictPct = ((telemetry?.memory_ratio_incl_dictionary ?? 0.184) - (telemetry?.memory_ratio_vs_fp16 ?? 0.181)) * 100;
  const fp16MB = ((telemetry?.fp16_bytes ?? 0) / 1024 / 1024);
  const compMB = ((telemetry?.compressed_bytes ?? 0) / 1024 / 1024);
  const compressionRatio = memPct > 0 ? (100 / memPct) : 5.5;
  const seqLen = telemetry?.seq_length ?? 0;

  const tierData = [
    { name: 'Local Window', value: telemetry?.per_tier_token_counts?.local_window_fp16 ?? (seqLen ? Math.min(seqLen, 64) : 0), color: '#10b981' },
    { name: 'Retrieval Heads', value: telemetry?.per_tier_token_counts?.retrieval_fp16 ?? 0, color: '#059669' },
    { name: 'H2O Peaked', value: telemetry?.per_tier_token_counts?.peaked_h2o_fp16 ?? 0, color: '#38bdf8' },
    { name: 'Lexico Sparse', value: telemetry?.per_tier_token_counts?.diffuse_lexico_sparse ?? 0, color: '#f59e0b' },
    { name: 'Evicted Tokens', value: telemetry?.per_tier_token_counts?.evicted_h2o ?? 0, color: '#e11d48' },
  ];

  return (
    <div className="bg-card border border-border rounded-xl p-4 shadow-sm space-y-4">
      {/* Title */}
      <div className="flex items-center justify-between border-b border-border/60 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-sm text-white">Live Compression Telemetry</h3>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-semibold ${
          seqLen > 0 ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-slate-700/50 text-gray-400'
        }`}>
          {seqLen > 0 ? `${seqLen.toLocaleString()} Tokens Live` : 'Standby · Ready'}
        </span>
      </div>

      {/* Memory Footprint Bar */}
      <div className="space-y-1.5 bg-bg/60 border border-border/70 rounded-lg p-3">
        <div className="flex justify-between items-baseline text-xs">
          <span className="text-gray-300 font-medium flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-sky-400" /> KV Memory vs FP16
          </span>
          <div className="flex items-baseline gap-1">
            <span className="font-mono text-base font-bold text-white">{memPct.toFixed(1)}%</span>
            <span className="text-[10px] text-emerald-400 font-mono font-semibold">({compressionRatio.toFixed(2)}× savings)</span>
          </div>
        </div>

        <div className="h-3 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-border">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-sky-400 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(Math.max(memPct, 4), 100)}%` }}
          />
        </div>

        <div className="flex justify-between text-[11px] text-gray-400 pt-0.5 font-mono">
          <span>Stored: {compMB > 0 ? `${compMB.toFixed(2)} MB` : '~18% target'}</span>
          <span>Baseline: {fp16MB > 0 ? `${fp16MB.toFixed(2)} MB` : '100% FP16'}</span>
          <span>Dict: +{dictPct.toFixed(1)}%</span>
        </div>
      </div>

      {/* 3 Metric Cards */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-bg/60 border border-border/60 rounded-lg p-2.5 text-center">
          <div className="text-[10px] text-gray-400 uppercase font-semibold">Active Atoms</div>
          <div className="text-base font-mono font-bold text-sky-300 mt-0.5">
            m={telemetry?.active_atoms ?? 16}
          </div>
          <div className="text-[10px] text-gray-500">of s={telemetry?.s_budget ?? 16} budget</div>
        </div>

        <div className="bg-bg/60 border border-border/60 rounded-lg p-2.5 text-center">
          <div className="text-[10px] text-gray-400 uppercase font-semibold">H2O Keep Ratio</div>
          <div className="text-base font-mono font-bold text-emerald-300 mt-0.5">
            {((telemetry?.keep_ratio ?? 0.15) * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-500">n_b=64 window</div>
        </div>

        <div className="bg-bg/60 border border-border/60 rounded-lg p-2.5 text-center">
          <div className="text-[10px] text-gray-400 uppercase font-semibold">HMM Shifts</div>
          <div className="text-base font-mono font-bold text-amber-300 mt-0.5">
            {telemetry?.hmm_shift_events ?? 0}
          </div>
          <div className="text-[10px] text-gray-500 font-mono">z={(telemetry?.hmm_last_z_score ?? 0).toFixed(2)}</div>
        </div>
      </div>

      {/* Per-Tier Distribution Breakdown */}
      <div className="bg-bg/60 border border-border/70 rounded-lg p-3 space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-300 font-medium">KV Cache Tier Distribution</span>
          <span className="text-[10px] text-gray-400 font-mono">
            {seqLen > 0 ? `${seqLen.toLocaleString()} context tokens` : 'Per-tier routing'}
          </span>
        </div>

        <div className="h-28 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tierData} layout="vertical" margin={{ top: 0, right: 10, left: 15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
              <YAxis dataKey="name" type="category" width={85} tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(value: number) => [`${value.toLocaleString()} tokens`, 'Tokens']}
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: '11px' }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {tierData.map((entry, index) => (
                  <Cell key={`tier-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
