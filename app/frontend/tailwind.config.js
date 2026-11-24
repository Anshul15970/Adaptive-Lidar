/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        lab: {
          bg: '#0a0c10',         // Deep graphite / charcoal base
          surface: '#101318',    // Workstation secondary surfaces / navigation
          panel: '#101318',      // Toolbars and side panels
          card: '#151921',       // Elevated containers
          cardHover: '#1c222c',  // Subtle hover state
          border: '#1f242e',     // Hairline engineering separator
          borderLight: '#2c3342',// Focused / active border
          textMuted: '#828c9b',  // Secondary descriptive text
          textDim: '#525c6c',    // Tertiary metadata labels
          textBright: '#f3f4f6', // Warm crisp white primary text
          accent: '#0ea5e9',     // Restrained technical electric blue
          accentMuted: '#0369a1',// Subdued blue
          cyan: '#0ea5e9',       // Map cyan to technical blue
          emerald: '#10b981',    // Muted automotive sage / green
          amber: '#f59e0b',      // Technical alert amber
          blue: '#3b82f6',       // Data series blue
          violet: '#8b5cf6',     // Auxiliary series
          rose: '#f43f5e',       // Baseline / error rose
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px -1px rgba(0, 0, 0, 0.4)',
        'workstation': '0 4px 16px -2px rgba(0, 0, 0, 0.5), 0 2px 6px -2px rgba(0, 0, 0, 0.4)',
      }
    },
  },
  plugins: [],
}
