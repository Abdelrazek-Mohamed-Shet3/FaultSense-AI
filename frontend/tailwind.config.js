/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#FFFFFF",
        alabaster: "#FAF9F6",
        stone: "#F3F1EB",
        border: "#E8E4DA",
        ink: {
          DEFAULT: "#1C1E1B",
          muted: "#5A5D55",
          subtle: "#8C8E85",
        },
        sage: {
          DEFAULT: "#4E7A65",
          hover: "#3F6452",
          soft: "#6E9582",
          tint: "#EFF5F1",
        },
        sand: {
          DEFAULT: "#B8886E",
          tint: "#F7F1ED",
        },
        amberwarm: {
          DEFAULT: "#B3833E",
          tint: "#FBF5EA",
        },
        terracotta: {
          DEFAULT: "#B85D4E",
          hover: "#9F4D40",
          tint: "#FDF0EE",
        },
        lavender: {
          DEFAULT: "#7E6C8F",
          tint: "#F4F1F7",
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', '"Cairo"', "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        soft: "0 2px 12px -2px rgba(28, 30, 27, 0.05), 0 1px 3px 0 rgba(28, 30, 27, 0.03)",
        card: "0 6px 24px -6px rgba(28, 30, 27, 0.06)",
      },
    },
  },
  plugins: [],
};
