export async function predictDefault(text: string) {
  const res = await fetch('/api/predict/default', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error('Prediction failed');
  return res.json();
}
