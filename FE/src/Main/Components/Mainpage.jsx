import Buttonbar from "./Buttonbar";
import Menubar from "./Menubar";
import background from "./assets/main_final.jpg";
import dialog from "./assets/dialog_frame.png";
import { MainStore } from "../Store";
import { useState } from "react";
import Rankingpage from "./Rankingpage";
import Mypage from "./Mypage";

const mainstyle = {
  width: "100%",
  minwidth: "260px",
  height: "100vh",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-between",
};

const backgroundstyle = {
  zIndex: "-1",
  position: "absolute",
  width: "100%",
  minwidth: "260px",
  top: "100px",
  height: "calc(100vh - 100px)",
};

const rankingpanelstyle = {
  zIndex: "2",
  position: "absolute",
  top: "20%",
  left: "3%",
  opacity: "1",
};

const dialogstyle = {
  display: "flex",
  justifyContent: "center",
  backgroundImage: `url(${dialog})`,
  position: "absolute",
  width: "340px",
  height: "250px",
  zIndex: "2",
  top: "38%",
  left: "38%",
  padding: "30px",
  transform: "scale(0) translateY(30%)",
  fontFamily: "비트비트체v2",
  fontSize: "24px",
  whiteSpace: "wrap",
};

const femaleNPC = {
  position: "absolute",
  top: "63%",
  left: "37%",
  width: "7%",
  height: "14%",
  cursor: "pointer",
};

const maleNPC = {
  position: "absolute",
  top: "63%",
  left: "62%",
  width: "7%",
  height: "14%",
  cursor: "pointer",
};

function Main() {
  const rankingpopup = MainStore((state) => state.rankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const dialogs = MainStore((state) => state.dialogs);

  const makedialog = () => {
    const index = Math.floor(Math.random() * dialogs.length);
    return dialogs[index];
  };

  const [text, setText] = useState("");
  const [textStyle, setTextStyle] = useState(dialogstyle);

  const NPCClick = () => {
    setText(makedialog());
    const start = setTimeout(() => {
      setTextStyle({
        ...textStyle,
        transform: "scale(0.5) translateY(0%)",
        transition: "all 0.2s",
      });
    }, 200);
    const end = setTimeout(() => {
      setTextStyle({ ...textStyle, transform: "scale(0) translateY(30%)" });
    }, 2200);
    return () => {
      clearTimeout(start);
      clearTimeout(end);
    };
  };

  return (
    <div style={mainstyle}>
      <Menubar
        data={{ image: "/loanplease.png", nickname: "ssafy", rank: 6 }}
      />
      <div style={rankingpanelstyle}>
        {rankingpopup ? <Rankingpage /> : <div></div>}
      </div>
      <div>{mypagepopup ? <Mypage /> : <div></div>}</div>
      <img style={backgroundstyle} src={background} />
      <div style={femaleNPC} onClick={NPCClick}></div>
      <div style={maleNPC} onClick={NPCClick}></div>
      <div style={textStyle}>{text}</div>
      <Buttonbar />
    </div>
  );
}

export default Main;
