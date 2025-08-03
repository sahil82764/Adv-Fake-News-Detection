import Link from 'next/link';

const MAX_CHARS = 2000;

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<{ label: string; confidence: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [charCount, setCharCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value.slice(0, MAX_CHARS);
    setText(value);
    setCharCount(value.length);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/predict/default`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error('Prediction failed');
      const data = await res.json();
      
      setResult({
        label: data.predictions[0],
        confidence: data.probabilities ? Math.round(data.probabilities[0] * 100) : 90,
      });
    } catch {
      setError('Prediction failed. Please try again.');
    }
    setLoading(false);
  };

  const getResultConfig = (label: string) => {
    if (label === 'Real') {
      return {
        title: 'Likely Genuine',
        description: 'This content shows characteristics of reliable information. The language is factual, sources appear credible, and claims can be verified.',
        icon: '✓',
        color: '#16a34a',
        bgColor: '#dcfce7',
        textColor: '#166534'
      };
    } else if (label === 'Fake') {
      return {
        title: 'Potentially Misleading',
        description: 'This content contains elements that suggest misinformation. Sensational language, unverified claims, or suspicious sources detected.',
        icon: '⚠',
        color: '#dc2626',
        bgColor: '#fef2f2',
        textColor: '#dc2626'
      };
    } else {
      return {
        title: 'Requires Verification',
        description: 'The analysis is inconclusive. This content needs additional fact-checking and source verification before determining credibility.',
        icon: '?',
        color: '#d97706',
        bgColor: '#fef3c7',
        textColor: '#d97706'
      };
    }
  };

  const getCharCountColor = () => {
    if (charCount > 1800) return '#dc2626';
    if (charCount > 1500) return '#d97706';
    return '#94a3b8';
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        *, *::before, *::after {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body, html, #__next {
          min-height: 100vh;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          position: relative;
          overflow-x: hidden;
        }

        .bg-animation {
          position: absolute;
          width: 100%;
          height: 100%;
          overflow: hidden;
          z-index: 0;
        }

        .floating-shape {
          position: absolute;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 50%;
          animation: float 6s ease-in-out infinite;
        }

        .shape-1 {
          width: 80px;
          height: 80px;
          top: 10%;
          left: 10%;
          animation-delay: 0s;
        }

        .shape-2 {
          width: 120px;
          height: 120px;
          top: 70%;
          right: 10%;
          animation-delay: 2s;
        }

        .shape-3 {
          width: 60px;
          height: 60px;
          bottom: 20%;
          left: 20%;
          animation-delay: 4s;
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }

        .container {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(20px);
          border-radius: 24px;
          padding: 40px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
          max-width: 600px;
          width: 100%;
          position: relative;
          z-index: 1;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
          text-align: center;
          margin-bottom: 32px;
        }

        .logo {
          display: inline-flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 16px;
        }

        .logo-icon {
          width: 48px;
          height: 48px;
          background: linear-gradient(45deg, #4f46e5, #7c3aed);
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 24px;
          font-weight: bold;
          animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }

        .logo-text {
          font-size: 28px;
          font-weight: 700;
          background: linear-gradient(45deg, #4f46e5, #7c3aed);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .tagline {
          color: #64748b;
          font-size: 16px;
          margin-bottom: 8px;
        }

        .subtitle {
          color: #475569;
          font-size: 14px;
          opacity: 0.8;
        }

        .input-section {
          position: relative;
          margin-bottom: 24px;
        }

        .input-wrapper {
          position: relative;
          background: white;
          border-radius: 16px;
          border: 2px solid #e2e8f0;
          transition: all 0.3s ease;
          overflow: hidden;
        }

        .input-wrapper:focus-within {
          border-color: #4f46e5;
          box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
          transform: translateY(-2px);
        }

        .input-field {
          width: 100%;
          padding: 20px;
          border: none;
          outline: none;
          font-size: 16px;
          font-family: inherit;
          resize: vertical;
          min-height: 120px;
          line-height: 1.5;
          color: #1e293b;
          background: transparent;
        }

        .input-field::placeholder {
          color: #94a3b8;
          font-style: italic;
        }

        .input-counter {
          position: absolute;
          bottom: 12px;
          right: 16px;
          font-size: 12px;
          color: ${getCharCountColor()};
          background: rgba(255, 255, 255, 0.9);
          padding: 4px 8px;
          border-radius: 8px;
        }

        .analyze-btn {
          width: 100%;
          background: linear-gradient(45deg, #4f46e5, #7c3aed);
          color: white;
          border: none;
          padding: 16px 24px;
          font-size: 16px;
          font-weight: 600;
          border-radius: 12px;
          cursor: pointer;
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
          margin-bottom: 24px;
        }

        .analyze-btn::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
          transition: left 0.5s ease;
        }

        .analyze-btn:hover::before {
          left: 100%;
        }

        .analyze-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3);
        }

        .analyze-btn:active {
          transform: translateY(0);
        }

        .analyze-btn:disabled {
          opacity: 0.8;
          cursor: not-allowed;
        }

        .btn-text {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }

        .loading-spinner {
          width: 20px;
          height: 20px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top: 2px solid white;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .result-section {
          background: #f8fafc;
          border-radius: 16px;
          padding: 24px;
          margin-top: 24px;
          border: 1px solid #e2e8f0;
          opacity: 0;
          transform: translateY(20px);
          transition: all 0.5s ease;
          animation: slideIn 0.5s ease forwards;
        }

        @keyframes slideIn {
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .result-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 16px;
        }

        .result-icon {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 16px;
          font-weight: bold;
        }

        .result-title {
          font-size: 18px;
          font-weight: 600;
          color: #1e293b;
        }

        .result-description {
          color: #475569;
          font-size: 14px;
          line-height: 1.5;
          margin-bottom: 16px;
        }

        .confidence-bar {
          background: #e2e8f0;
          height: 8px;
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 8px;
        }

        .confidence-fill {
          height: 100%;
          border-radius: 4px;
          transition: width 1s ease;
        }

        .confidence-text {
          font-size: 12px;
          color: #64748b;
          text-align: center;
        }

        .tips {
          background: rgba(79, 70, 229, 0.05);
          border-radius: 12px;
          padding: 16px;
          border-left: 4px solid #4f46e5;
          margin-top: 24px;
        }

        .tips h4 {
          color: #4f46e5;
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 8px;
        }

        .tips p {
          color: #475569;
          font-size: 13px;
          line-height: 1.4;
        }

        .error-section {
          background: #fef2f2;
          border: 1px solid #fecaca;
          border-radius: 16px;
          padding: 24px;
          margin-top: 24px;
          text-align: center;
          color: #dc2626;
          font-weight: 600;
          animation: slideIn 0.5s ease forwards;
        }

        @media (max-width: 640px) {
          .container {
            padding: 24px;
            margin: 10px;
          }
          
          .logo-text {
            font-size: 24px;
          }
        }
      `}</style>

      <div className="bg-animation">
        <div className="floating-shape shape-1"></div>
        <div className="floating-shape shape-2"></div>
        <div className="floating-shape shape-3"></div>
      </div>

      <div className="container">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
          <Link href="/compare" style={{
            color: '#7c3aed',
            fontWeight: 600,
            textDecoration: 'none',
            fontSize: 15,
            background: 'rgba(124,58,237,0.08)',
            padding: '6px 16px',
            borderRadius: 8,
            transition: 'background 0.2s',
            marginRight: 0
          }}>
            Compare Models
          </Link>
          <Link href="/analytics" style={{
            color: '#4f46e5',
            fontWeight: 600,
            textDecoration: 'none',
            fontSize: 15,
            background: 'rgba(79,70,229,0.08)',
            padding: '6px 16px',
            borderRadius: 8,
            transition: 'background 0.2s',
            marginLeft: 0,
            marginRight: 0
          }}>
            Analytics
          </Link>
          <Link href="/help" style={{
            color: '#10b981',
            fontWeight: 600,
            textDecoration: 'none',
            fontSize: 15,
            background: 'rgba(16,185,129,0.08)',
            padding: '6px 16px',
            borderRadius: 8,
            transition: 'background 0.2s',
            marginLeft: 0
          }}>
            Help
          </Link>
        </div>

        <div className="header">
          <div className="logo">
            <div className="logo-icon">🔍</div>
            <div className="logo-text">Fake News Detection System</div>
          </div>
          <div className="tagline">AI-Powered Fact Verification</div>
          <div className="subtitle">Analyze text for misinformation and verify credibility</div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-section">
            <div className="input-wrapper">
              <textarea
                className="input-field"
                value={text}
                onChange={handleInput}
                placeholder="Paste or type the news article, social media post, or any text you want to verify for accuracy..."
                maxLength={MAX_CHARS}
                required
              />
              <div className="input-counter">
                <span>{charCount}</span>/{MAX_CHARS}
              </div>
            </div>
          </div>

          <button
            className="analyze-btn"
            type="submit"
            disabled={loading || !text.trim()}
          >
            <div className="btn-text">
              {loading && <div className="loading-spinner"></div>}
              <span>🔍 Analyze for Misinformation</span>
            </div>
          </button>
        </form>

        {error && (
          <div className="error-section">
            {error}
          </div>
        )}

        {result && !error && (() => {
          const config = getResultConfig(result.label);
          return (
            <div className="result-section">
              <div className="result-header">
                <div
                  className="result-icon"
                  style={{
                    backgroundColor: config.bgColor,
                    color: config.textColor
                  }}
                >
                  {config.icon}
                </div>
                <div>
                  <div className="result-title">{config.title}</div>
                </div>
              </div>
              <div className="result-description">
                {config.description}
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${result.confidence}%`,
                    backgroundColor: config.color
                  }}
                />
              </div>
              <div className="confidence-text">
                Confidence: {result.confidence}%
              </div>
            </div>
          );
        })()}

        <div className="tips">
          <h4>💡 Pro Tips for Fact-Checking</h4>
          <p>
            Always cross-reference with multiple sources, check the publication date, 
            verify author credentials, and look for supporting evidence before sharing information.
          </p>
        </div>
      </div>
    </>
  );
}