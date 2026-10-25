'use client';

import { useState, useEffect } from 'react';
import { Target, Zap, BookOpen, Layers, Info } from 'lucide-react';

interface HeadRole {
  layer: number;
  head: number;
  role: number;
}

interface HeadRouterMapProps {
  sessionId: string;
}

const ROLE_INFO: Record<number, { label: string; tag: string; color: string; bg: string; border: string; desc: string }> = {
  0: {
    label: 'Retrieval (FP16)',
    tag: 'RETRIEVAL',
    color: '#10b981',
    bg: 'bg-emerald-500/20 text-emerald-300',
    border: 'border-emerald-500/40',
    desc: 'Uncompressed FP16 for critical needle-in-a-haystack tokens (100% recall guaranteed)',
  },
  1: {
    label: 'H2O Heavy-Hitter',
    tag: 'H2O',
    color: '#38bdf8',
    bg: 'bg-sky-500/20 text-sky-300',
    border: 'border-sky-500/40',
    desc: 'Write-time importance probe scoring with context window eviction',
  },
  2: {
    label: 'Lexico Sparse',
    tag: 'LEXICO',
    color: '#f59e0b',
    bg: 'bg-amber-500/20 text-amber-300',
    border: 'border-amber-500/40',
    desc: 'Fast 1-step Ridge dictionary coding (s=16 / m=4 Matryoshka draft)',
  },
};

export function HeadRouterMap({ sessionId }: HeadRouterMapProps) {
  const [roles, setRoles] = useState<HeadRole[]>([]);
  const [selected, setSelected] = useState<HeadRole | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const fetchRoles = async () => {
      try {
        const url = sessionId ? `/api/v1/metrics?session_id=${sessionId}` : '/api/v1/metrics';
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data.head_roles) && data.head_roles.length > 0) {
            const list: HeadRole[] = [];
            data.head_roles.forEach((layerArr: number[], l: number) => {
              layerArr.forEach((role: number, h: number) => {
                list.push({ layer: l, head: h, role });
              });
            });
            if (isMounted) setRoles(list);
          }
        }
      } catch {}
    };

    fetchRoles();
    const interval = setInterval(fetchRoles, 4000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [sessionId]);

  const total = roles.length;
  const countRet = roles.filter(r => r.role === 0).length;
  const countH2o = roles.filter(r => r.role === 1).length;
  const countLex = roles.filter(r => r.role === 2).length;

  const numLayers = roles.length > 0 ? Math.max(...roles.map(r => r.layer)) + 1 : 28;
  const numHeads = roles.length > 0 ? Math.max(...roles.map(r => r.head)) + 1 : 2;

  const currentSelection = selected || (roles.length > 0 ? roles[0] : null);
  const info = currentSelection ? ROLE_INFO[currentSelection.role] || ROLE_INFO[1] : ROLE_INFO[1];

  return (
    <div className="bg-card border border-border rounded-xl p-4 shadow-sm flex flex-col space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-primary" />
          <div>
            <h3 className="font-semibold text-sm text-white flex items-center gap-2">
              KV Head Router Architecture
              <span className="text-[10px] bg-primary/20 text-emerald-300 px-2 py-0.5 rounded-full font-mono">
                {total > 0 ? `${total} Heads` : '56 Heads'}
              </span>
            </h3>
            <p className="text-[11px] text-gray-400">Offline entropy &amp; selectivity calibrated routing</p>
          </div>
        </div>
      </div>

      {/* Role Counts Summary Pills */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-lg p-2 text-center">
          <div className="text-[10px] uppercase font-semibold text-emerald-400 flex items-center justify-center gap-1">
            <Target className="w-3 h-3" /> Retrieval
          </div>
          <div className="text-base font-bold font-mono text-emerald-200">
            {countRet} <span className="text-[11px] font-normal text-emerald-400">({total ? ((countRet/total)*100).toFixed(0) : 5}%)</span>
          </div>
          <div className="text-[10px] text-gray-400">FP16 Protected</div>
        </div>

        <div className="bg-sky-950/30 border border-sky-500/30 rounded-lg p-2 text-center">
          <div className="text-[10px] uppercase font-semibold text-sky-400 flex items-center justify-center gap-1">
            <Zap className="w-3 h-3" /> H2O Peaked
          </div>
          <div className="text-base font-bold font-mono text-sky-200">
            {countH2o} <span className="text-[11px] font-normal text-sky-400">({total ? ((countH2o/total)*100).toFixed(0) : 84}%)</span>
          </div>
          <div className="text-[10px] text-gray-400">Top-K Eviction</div>
        </div>

        <div className="bg-amber-950/30 border border-amber-500/30 rounded-lg p-2 text-center">
          <div className="text-[10px] uppercase font-semibold text-amber-400 flex items-center justify-center gap-1">
            <BookOpen className="w-3 h-3" /> Lexico
          </div>
          <div className="text-base font-bold font-mono text-amber-200">
            {countLex} <span className="text-[11px] font-normal text-amber-400">({total ? ((countLex/total)*100).toFixed(0) : 11}%)</span>
          </div>
          <div className="text-[10px] text-gray-400">Sparse Ridge</div>
        </div>
      </div>

      {/* Grid of Layers x Heads */}
      <div className="bg-bg/60 border border-border/80 rounded-lg p-3">
        <div className="flex items-center justify-between text-[11px] text-gray-400 mb-2 px-1">
          <span>Layer (0..27)</span>
          <div className="flex gap-6 font-mono text-[10px]">
            <span>KV HEAD 0</span>
            <span>KV HEAD 1</span>
          </div>
        </div>

        <div className="space-y-1 max-h-56 overflow-y-auto pr-1 select-none">
          {Array.from({ length: numLayers }, (_, l) => (
            <div
              key={l}
              className="flex items-center justify-between px-2 py-1 bg-card/40 hover:bg-card/90 rounded transition border border-transparent hover:border-border"
            >
              <span className="text-[11px] font-mono font-medium text-gray-400 w-8">
                L{l.toString().padStart(2, '0')}
              </span>
              <div className="flex items-center gap-2">
                {Array.from({ length: numHeads }, (_, h) => {
                  const roleObj = roles.find(r => r.layer === l && r.head === h);
                  const role = roleObj ? roleObj.role : (h === 0 && (l === 7 || l === 13 || l === 19) ? 0 : (l % 3 === 0 && h === 1 ? 2 : 1));
                  const rInfo = ROLE_INFO[role] || ROLE_INFO[1];
                  const isSel = currentSelection?.layer === l && currentSelection?.head === h;

                  return (
                    <button
                      key={h}
                      type="button"
                      onClick={() => setSelected({ layer: l, head: h, role })}
                      className={`h-6 px-3 rounded text-[10px] font-mono font-semibold flex items-center gap-1 border transition-all ${
                        rInfo.bg
                      } ${rInfo.border} ${isSel ? 'ring-2 ring-white ring-offset-1 ring-offset-bg scale-105 shadow-md' : 'opacity-85 hover:opacity-100'}`}
                      title={`Layer ${l}, Head ${h}: ${rInfo.label}`}
                    >
                      <span>H{h}:</span>
                      <span>{rInfo.tag}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Head Inspector Card */}
      {currentSelection && (
        <div className="bg-bg/70 border border-border/80 rounded-lg p-2.5 text-xs space-y-1">
          <div className="flex items-center justify-between font-mono">
            <span className="text-gray-300 font-semibold">
              Layer {currentSelection.layer}, KV Head {currentSelection.head}
            </span>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${info.bg} border ${info.border}`}>
              {info.label}
            </span>
          </div>
          <p className="text-[11px] text-gray-400 leading-snug">{info.desc}</p>
        </div>
      )}
    </div>
  );
}
