/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#060c1a',
          900: '#0b1426',
          800: '#111f38',
          700: '#1a2e50',
        },
      },
    },
  },
  plugins: [],
}
