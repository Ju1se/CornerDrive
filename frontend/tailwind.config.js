/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Layer colors
        'l0': '#6b7280', // gray (out of scope)
        'l1': '#3b82f6', // blue
        'l2': '#f97316', // orange
        'l3': '#22c55e', // green
        'l4': '#a855f7', // purple
        // Classification colors
        'fraud': '#ef4444',
        'rarity': '#22c55e',
        'noise': '#6b7280',
        'honest': '#3b82f6',
      },
    },
  },
  plugins: [],
};