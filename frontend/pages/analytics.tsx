import React, { useEffect, useState } from 'react';

function ConfusionMatrix({ matrix, labels }: { matrix: number[][], labels: string[] }) {
  return (
    <table style={{ margin: '12px auto', borderCollapse: 'collapse', background: '#f9fafb', borderRadius: 8 }}>
      <thead>
        <tr>
          <th style={{ padding: 6, border: '1px solid #e5e7eb', background: '#f3f4f6' }}></th>
          {labels.map(l => (
            <th key={l} style={{ padding: 6, border: '1px solid #e5e7eb', background: '#f3f4f6' }}>{l}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <th style={{ padding: 6, border: '1px solid #e5e7eb', background: '#f3f4f6' }}>{labels[i]}</th>
            {row.map((val, j) => (
              <td key={j} style={{ padding: 6, border: '1px solid #e5e7eb', fontWeight: 600 }}>{val}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Analytics() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMetrics() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/compare/performance');
        if (!res.ok) throw new Error('Failed to fetch metrics');
        setMetrics(await res.json());
      } catch {
        setError('Failed to load analytics.');
      }
      setLoading(false);
    }
    fetchMetrics();
  }, []);

  const getColor = (val: number, type: string) => {
    if (type === 'f1_score' || type === 'accuracy') {
      if (val > 0.9) return '#16a34a';
      if (val > 0.8) return '#d97706';
      return '#dc2626';
    }
    return '#6366f1';
  };

  return (
    <div style={{ maxWidth: 700, margin: '40px auto', padding: 24, background: 'rgba(255,255,255,0.97)', borderRadius: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
      <h2 style={{ textAlign: 'center', marginBottom: 24, color: '#7c3aed' }}>Model Performance Analytics</h2>
      {loading && <div style={{ textAlign: 'center', color: '#7c3aed' }}>Loading analytics...</div>}
      {error && <div style={{ color: '#dc2626', fontWeight: 600, textAlign: 'center' }}>{error}</div>}
      {metrics && (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15, marginTop: 16 }}>
          <thead>
            <tr style={{ background: '#f3f4f6' }}>
              <th style={{ padding: 8, borderRadius: 8 }}>Model <span title="Click model name for confusion matrix" style={{cursor:'help',color:'#6366f1'}}>?</span></th>
              <th style={{ padding: 8, borderRadius: 8 }}>F1 Score <span title="Harmonic mean of precision and recall"></span></th>
              <th style={{ padding: 8, borderRadius: 8 }}>Accuracy <span title="Correct predictions / total"></span></th>
              <th style={{ padding: 8, borderRadius: 8 }}>Precision <span title="TP / (TP + FP)"></span></th>
              <th style={{ padding: 8, borderRadius: 8 }}>Recall <span title="TP / (TP + FN)"></span></th>
              <th style={{ padding: 8, borderRadius: 8 }}>Avg Inference (ms) <span title="Average time per prediction"></span></th>
              <th style={{ padding: 8, borderRadius: 8 }}>Memory (MB) <span title="Peak memory usage"></span></th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([model, data]: any) => (
              <React.Fragment key={model}>
                <tr style={{ background: '#fff', borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: 8, cursor: 'pointer', color: '#4f46e5', fontWeight: 600 }}
                    onClick={() => setExpanded(expanded === model ? null : model)}
                    title="Click to show/hide confusion matrix"
                  >
                    {model}
                  </td>
                  <td style={{ padding: 8, color: getColor(data.performance?.f1_score, 'f1_score'), fontWeight: 600 }}>{data.performance?.f1_score?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: 8, color: getColor(data.performance?.accuracy, 'accuracy'), fontWeight: 600 }}>{data.performance?.accuracy?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: 8 }}>{data.performance?.precision?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: 8 }}>{data.performance?.recall?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: 8 }}>{data.resources?.avg_inference_ms ?? '-'}</td>
                  <td style={{ padding: 8 }}>{data.resources?.memory_mb ?? '-'}</td>
                </tr>
                {expanded === model && data.performance?.confusion_matrix && (
                  <tr>
                    <td colSpan={7}>
                      <div style={{ margin: '8px 0 16px 0', textAlign: 'center' }}>
                        <strong>Confusion Matrix</strong>
                        <ConfusionMatrix matrix={data.performance.confusion_matrix.matrix} labels={data.performance.confusion_matrix.labels} />
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
