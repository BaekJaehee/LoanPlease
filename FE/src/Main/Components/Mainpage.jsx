import Buttonbar from "./Buttonbar";
import Menubar from "./Menubar";
import background from "./assets/main_final.jpg";
import { MainStore } from "../Store";
import { useState } from "react";
import Rankingpage from "./Rankingpage";
import Mypage from "./Mypage";

const mainstyleClass = `
z-[1] w-full min-w-[260px] h-screen 
flex flex-col justify-between
`;

const backgroundstyleClass = `
absolute z-[0] 
w-full min-w-[260px]
top-[100px] h-[calc(100vh_-_100px)] 
`;

const rankingpanelstyleClass = `
absolute z-[2] 
top-[20%] left-[3%] opacity-100 
`;

const mypagepanelstyleClass = `
flex justify-center z-[2]
`;

const dialogstyleClass = `
flex justify-center absolute 
w-[340px] h-[250px] z-[5] 
top-[38%] left-[38%] p-[30px] 
bg-npcLeft bg-cover 
scale-0 translate-y-[30%] 
font-cusFont1 text-2xl whitespace-pre-wrap 
`;

const femaleNPCClass = `
absolute w-[7%] h-[14%] 
top-[63%] left-[37%] cursor-pointer 
`;

const maleNPCClass = `
absolute w-[7%] h-[14%] 
top-[63%] left-[62%] cursor-pointer
`;

// const mainstyle = {
//   width: "100%",
//   minwidth: "260px",
//   height: "100vh",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "space-between",
// };

// const backgroundstyle = {
//   zIndex: "-1",
//   position: "absolute",
//   width: "100%",
//   minwidth: "260px",
//   top: "100px",
//   height: "calc(100vh - 100px)",
// };

// const rankingpanelstyle = {
//   zIndex: "2",
//   position: "absolute",
//   top: "20%",
//   left: "3%",
//   opacity: "1",
// };

// const dialogstyle = {
//   display: "flex",
//   justifyContent: "center",
//   backgroundImage: `url(${dialog})`,
//   position: "absolute",
//   width: "340px",
//   height: "250px",
//   zIndex: "2",
//   top: "38%",
//   left: "38%",
//   padding: "30px",
//   transform: "scale(0) translateY(30%)",
//   fontFamily: "비트비트체v2",
//   fontSize: "24px",
//   whiteSpace: "wrap",
// };

// const femaleNPC = {
//   position: "absolute",
//   top: "63%",
//   left: "37%",
//   width: "7%",
//   height: "14%",
//   cursor: "pointer",
// };

// const maleNPC = {
//   position: "absolute",
//   top: "63%",
//   left: "62%",
//   width: "7%",
//   height: "14%",
//   cursor: "pointer",
// };

function Main() {
  const rankingpopup = MainStore((state) => state.rankingpopup);
  const setRankingpopup = MainStore((state) => state.setRankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const setMypagepopup = MainStore((state) => state.setMypagepopup);
  const dialogs = MainStore((state) => state.dialogs);

  const makedialog = () => {
    const index = Math.floor(Math.random() * dialogs.length);
    return dialogs[index];
  };

  const [text, setText] = useState("");
  // const [textStyle, setTextStyle] = useState(dialogstyle);
  const [textStyle, setTextStyle] = useState(dialogstyleClass);

  const NPCClick = () => {
    setText(makedialog());
    const start = setTimeout(() => {
      // setTextStyle({
      //   ...textStyle,
      //   transform: "scale(0.5) translateY(0%)",
      //   transition: "all 0.2s",
      // });
      setTextStyle(
        textStyle.replace(
          "scale-0 translate-y-[30%]",
          "scale-50 translate-y-[0%]",
        ),
      );
      if (!textStyle.includes(" transition-all duration-200")) {
        setTextStyle(textStyle + " transition-all duration-200");
      }
    }, 200);
    const end = setTimeout(() => {
      // setTextStyle({ ...textStyle, transform: "scale(0) translateY(30%)" });
      setTextStyle(
        textStyle.replace(
          "scale-50 translate-y-[0%]",
          "scale-0 translate-y-[30%]",
        ),
      );
    }, 2200);
    return () => {
      clearTimeout(start);
      clearTimeout(end);
    };
  };

  return (
    // <div style={mainstyle}>
    <div className={mainstyleClass}>
      <Menubar
        data={{ image: "/loanplease.png", nickname: "ssafy", rank: 6 }}
      />
      {/* <div style={rankingpanelstyle}> */}
      <div className={rankingpanelstyleClass}>
        {rankingpopup ? <Rankingpage /> : <div></div>}
      </div>
      <div className={mypagepanelstyleClass}>
        {mypagepopup ? <Mypage /> : <div></div>}
      </div>
      {/* <img style={backgroundstyle} src={background} /> */}
      <img
        onClick={() => {
          setRankingpopup(false);
          setMypagepopup(false);
        }}
        className={backgroundstyleClass}
        src={background}
      />
      {/* <div style={femaleNPC} onClick={NPCClick}></div> */}
      <div
        className={femaleNPCClass}
        onClick={() => {
          if (!textStyle.includes("bg-npcLeft")) {
            setTextStyle(textStyle.replace("bg-npcRight", "bg-npcLeft"));
          }
          NPCClick();
        }}
      ></div>
      {/* <div style={maleNPC} onClick={NPCClick}></div> */}
      <div
        className={maleNPCClass}
        onClick={() => {
          if (!textStyle.includes("bg-npcRight")) {
            setTextStyle(textStyle.replace("bg-npcLeft", "bg-npcRight"));
          }
          NPCClick();
        }}
      ></div>
      {/* <div style={textStyle}>{text}</div> */}
      {/* <div style={textStyle}>{text}</div> */}
      <div className={textStyle}>{text}</div>
      <Buttonbar />
    </div>
  );
}

export default Main;
