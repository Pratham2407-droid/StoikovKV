export interface GenerateRequest {
  prompt: string;
  session_id?: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  use_matryoshka?: boolean;
  draft_atoms?: number;
  verify_atoms?: number;
  gamma_spec?: number;
}

export interface GenerateResponse {
  session_id: string;
  type: 'start' | 'token' | 'done' | 'error';
  token?: number;
  text?: string;
  metrics?: any;
  message?: string;
}

export async function* streamGenerate(request: GenerateRequest): AsyncGenerator<GenerateResponse> {
  const response = await fetch('/api/v1/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Generation failed: ${response.status}`);
  }

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
            yield JSON.parse(line.slice(6));
          } catch {}
        }
      }
    }
  }
}

export async function getMetrics(sessionId: string) {
  const res = await fetch(`/api/v1/metrics?session_id=${sessionId}`);
  return res.json();
}

export async function healthCheck() {
  const res = await fetch('/api/v1/health');
  return res.json();
}