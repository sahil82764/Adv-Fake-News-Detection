import React from 'react';

export function Button({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      style={{
        background: '#2563eb',
        color: '#fff',
        border: 'none',
        borderRadius: 4,
        padding: '8px 20px',
        fontWeight: 600,
        fontSize: 16,
        cursor: props.disabled ? 'not-allowed' : 'pointer',
        opacity: props.disabled ? 0.6 : 1,
        marginTop: 8,
        ...props.style,
      }}
    >
      {children}
    </button>
  );
}
