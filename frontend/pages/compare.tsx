import React, { useState } from 'react';

const MODELS = [
  { key: 'distilbert', label: 'DistilBERT (Transformer)' },
  { key: 'albert', label: 'ALBERT (Transformer)' },
  { key: 'logistic_regression', label: 'Logistic Regression (Traditional)' },
  { key: 'naive_bayes', label: 'Naive Bayes (Traditional)' },
  { key: 'svm', label: 'SVM (Traditional)' },
];

export default function ModelComparison() {
  const [text, setText] = useState('');
  const [selected, setSelected] = useState<string[]>(['distilbert']);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleModelChange = (key: string) => {
    setSelected(sel =>
      sel.includes(key) ? sel.filter(k => k !== key) : [...sel, key]
    );
    setResults(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResults(null);
    setError(null);
    try {
      const res = await fetch('/api/compare/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, models: selected }),
      });
      if (!res.ok) throw new Error('Comparison failed');
      setResults(await res.json());
    } catch {
      setError('Comparison failed. Please try again.');
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 700, margin: '40px auto', padding: 24, background: 'rgba(255,255,255,0.97)', borderRadius: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
      <h2 style={{ textAlign: 'center', marginBottom: 24, color: '#7c3aed' }}>Compare Multiple Models</h2>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste news article here..."
          rows={4}
          style={{ width: '100%', borderRadius: 16, padding: 16, fontSize: 16, border: '1.5px solid #e5e7eb', background: '#f9fafb' }}
          required
        />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 8 }}>
          {MODELS.map(m => (
            <label key={m.key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 15 }}>
              <input
                type="checkbox"
                checked={selected.includes(m.key)}
                onChange={() => handleModelChange(m.key)}
                style={{ accentColor: '#7c3aed', width: 18, height: 18 }}
              />
              {m.label}
            </label>
          ))}
        </div>
        <button
          type="submit"
          disabled={loading || !text.trim() || selected.length === 0}
          style={{
            width: '100%',
            background: 'linear-gradient(90deg, #4f46e5, #7c3aed)',
            color: '#fff',
            border: 'none',
            borderRadius: 12,
            padding: '14px 0',
            fontWeight: 600,
            fontSize: 16,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1,
            marginTop: 8,
            boxShadow: '0 2px 8px #7c3aed22',
            transition: 'transform 0.15s, box-shadow 0.15s',
          }}
        >
          {loading ? 'Comparing...' : '🔍 Compare Selected Models'}
        </button>
      </form>
      {error && <div style={{ color: '#dc2626', fontWeight: 600, marginTop: 16 }}>{error}</div>}
      {results && (
        <div style={{ marginTop: 32 }}>
          <h3 style={{ textAlign: 'center', marginBottom: 12, color: '#4f46e5' }}>Results</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15 }}>
            <thead>
              <tr style={{ background: '#f3f4f6' }}>
                <th style={{ padding: 8, borderRadius: 8 }}>Model</th>
                <th style={{ padding: 8, borderRadius: 8 }}>Prediction</th>
                <th style={{ padding: 8, borderRadius: 8 }}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results).map(([model, res]: any) => (
                <tr key={model} style={{ background: '#fff', borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: 8 }}>{model}</td>
                  <td style={{ padding: 8, color: res.prediction === 'Real' ? '#16a34a' : res.prediction === 'Fake' ? '#dc2626' : '#d97706', fontWeight: 600 }}>{res.prediction}</td>
                  <td style={{ padding: 8 }}>{res.probability !== undefined ? `${Math.round(res.probability * 100)}%` : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
