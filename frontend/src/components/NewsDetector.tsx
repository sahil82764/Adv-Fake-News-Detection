import React, { useState } from 'react';
import { predictDefault } from '../lib/api';
import { Button } from './ui/button';

export default function NewsDetector() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await predictDefault(text);
      setResult(data.predictions[0]);
    } catch (err) {
      setError('Prediction failed. Please try again.');
    }
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Paste news article here..."
        rows={4}
        style={{
          width: '100%',
          padding: 12,
          border: '1px solid #ccc',
          borderRadius: 4,
          fontSize: 16,
          outline: 'none',
          boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
          transition: 'border 0.2s',
        }}
        onFocus={e => (e.currentTarget.style.border = '1.5px solid #2563eb')}
        onBlur={e => (e.currentTarget.style.border = '1px solid #ccc')}
      />
      <Button type="submit" disabled={loading || !text}>
        {loading ? 'Predicting...' : 'Detect Fake News'}
      </Button>
      {result && !error && (
        <div style={{ color: result === 'Fake' ? '#dc2626' : '#16a34a', fontWeight: 600, marginTop: 8 }}>
          Prediction: <b>{result}</b>
        </div>
      )}
      {error && (
        <div style={{ color: '#dc2626', fontWeight: 600, marginTop: 8 }}>
          Prediction: <b>Error: {error}</b>
        </div>
      )}
    </form>
  );
}
