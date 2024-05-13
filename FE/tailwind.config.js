/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        cusFont1: ["비트비트체v2"], // 비트비트체v2
        cusFont2: ["Orbit"], // Orbit
      },
      colors: {
        cusColor1: "#186F65", // 진초록색
        cusColor2: "#795458", // 짙은 갈색
        cusColor3: "#FFC94A", // 진노란색
        cusColor4: "#FFFAB7", // 연노란색
        cusColor5: "#5BBCFF", // 하늘색
      },
      backgroundImage: {
        follow: "url('./Main/Components/assets/follow.png')",
        unfollow: "url('./Main/Components/assets/unfollow.png')",
        npcLeft: "url('./Main/Components/assets/dialog_frame.png')",
        npcRight: "url('./Main/Components/assets/dialog_frame_reversed.png')",
        game: "url('./Game/Assets/background.png')",
        speechBubble: "url('./Game/Assets/speech_bubble.png')",
      },
    },
  },
  plugins: [],
};
