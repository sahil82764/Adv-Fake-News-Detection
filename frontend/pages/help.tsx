import React from 'react';

const MODEL_INFO = [
	{
		name: 'DistilBERT (Transformer)',
		desc: 'A lightweight transformer model for natural language understanding, fine-tuned for fake news detection.',
	},
	{
		name: 'ALBERT (Transformer)',
		desc: 'A lite version of BERT with parameter sharing, offering efficient and accurate text classification.',
	},
	{
		name: 'Logistic Regression (Traditional)',
		desc: 'A classic linear model for binary classification, using TF-IDF features from news text.',
	},
	{
		name: 'Naive Bayes (Traditional)',
		desc: 'A probabilistic model based on Bayes theorem, effective for text classification tasks.',
	},
	{
		name: 'SVM (Traditional)',
		desc: 'Support Vector Machine, a robust classifier that finds the optimal boundary between classes.',
	},
];

const METRIC_INFO = [
	{ name: 'Accuracy', desc: 'Proportion of correct predictions out of all predictions.' },
	{ name: 'F1 Score', desc: 'Harmonic mean of precision and recall, balances false positives and negatives.' },
	{ name: 'Precision', desc: 'Proportion of positive identifications that were actually correct.' },
	{ name: 'Recall', desc: 'Proportion of actual positives that were identified correctly.' },
	{ name: 'Confusion Matrix', desc: 'Table showing correct and incorrect predictions for each class.' },
	{ name: 'ROC Curve', desc: 'Graph showing the trade-off between true positive and false positive rates.' },
	{ name: 'Avg Inference (ms)', desc: 'Average time taken for the model to make a prediction.' },
	{ name: 'Memory (MB)', desc: 'Peak memory usage during inference.' },
];

export default function HelpPage() {
	return (
		<div
			style={{
				maxWidth: 700,
				margin: '40px auto',
				padding: 24,
				background: 'rgba(255,255,255,0.97)',
				borderRadius: 24,
				boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
			}}
		>
			<h2
				style={{
					textAlign: 'center',
					marginBottom: 24,
					color: '#7c3aed',
				}}
			>
				Help & About
			</h2>
			<section style={{ marginBottom: 32 }}>
				<h3
					style={{
						color: '#4f46e5',
						marginBottom: 12,
					}}
				>
					Model Types
				</h3>
				<ul style={{ paddingLeft: 20 }}>
					{MODEL_INFO.map((m) => (
						<li key={m.name} style={{ marginBottom: 10 }}>
							<strong>{m.name}:</strong> {m.desc}
						</li>
					))}
				</ul>
			</section>
			<section>
				<h3
					style={{
						color: '#4f46e5',
						marginBottom: 12,
					}}
				>
					Evaluation Metrics
				</h3>
				<ul style={{ paddingLeft: 20 }}>
					{METRIC_INFO.map((m) => (
						<li key={m.name} style={{ marginBottom: 10 }}>
							<strong>{m.name}:</strong> {m.desc}
						</li>
					))}
				</ul>
			</section>
			<div
				style={{
					marginTop: 32,
					color: '#6b7280',
					fontSize: 15,
					textAlign: 'center',
				}}
			>
				For more details, see the{' '}
				<a
					href="/analytics"
					style={{
						color: '#4f46e5',
						textDecoration: 'underline',
					}}
				>
					Analytics
				</a>{' '}
				page or{' '}
				<a
					href="/compare"
					style={{
						color: '#4f46e5',
						textDecoration: 'underline',
					}}
				>
					Compare Models
				</a>
				.
			</div>
		</div>
	);
}
