/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0f172a',
        card: '#1e293b',
        border: '#334155',
        primary: '#059669',
        secondary: '#0284c7',
        warning: '#d97706',
        danger: '#e11d48',
      },
    },
  },
  plugins: [],
};