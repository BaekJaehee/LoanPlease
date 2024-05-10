import Buttonbar from "./Buttonbar";
import Menubar from "./Menubar";
import background from "./assets/main_final.jpg";
import { LoginStore, MainStore } from "../Store";
import { useEffect, useState } from "react";
import Rankingpage from "./Rankingpage";
import Mypage from "./Mypage";
import Noticebar from "./Noticebar";

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
top-[20%] left-[3%] 
`;

const mypagepanelstyleClass = `
flex justify-center z-[2]
`;

const noticeareastyleClass = `
absolute translate-x-[calc(100vw_-_350px)] translate-y-[20vh] overflow-y-auto 
w-[350px] max-h-[calc(100vh_-_20%)] z-20
`;

const dialogstyleClass = `
absolute w-[340px] h-[250px] z-[5] 
top-[38%] left-[38%] 
scale-0 translate-y-[30%] 
font-cusFont1 text-2xl whitespace-pre-wrap 
`;

const dialogtextstyleClass = `
flex justify-center 
w-full h-full p-[30px]  
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

// 친구 알림 테스트용 데이터
const noticedata = [];
// const noticedata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

function Main() {
  const rankingpopup = MainStore((state) => state.rankingpopup);
  const setRankingpopup = MainStore((state) => state.setRankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const setMypagepopup = MainStore((state) => state.setMypagepopup);
  const mydata = LoginStore((state) => state.mydata);
  const setMyData = LoginStore((state) => state.setMyData);
  const setIsBgm = MainStore((state) => state.setIsBgm);
  const dialogs = MainStore((state) => state.dialogs);

  const makedialog = () => {
    const index = Math.floor(Math.random() * dialogs.length);
    return dialogs[index];
  };

  const [text, setText] = useState("");
  // const [textStyle, setTextStyle] = useState(dialogstyle);
  const [textStyle, setTextStyle] = useState(dialogstyleClass);
  const [imgStyle, setImgStyle] = useState(
    dialogtextstyleClass + " bg-npcLeft bg-cover",
  );

  const makeClickSound = () => {
    const audio = new Audio("audioes/pop_sound_2.mp3");
    audio.play();
  };

  const NPCClick = () => {
    makeClickSound();
    setText(makedialog());
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
    const end = setTimeout(() => {
      // setTextStyle({ ...textStyle, transform: "scale(0) translateY(30%)" });
      setTextStyle(
        textStyle.replace(
          "scale-50 translate-y-[0%]",
          "scale-0 translate-y-[30%]",
        ),
      );
    }, 2000);
    return () => {
      clearTimeout(end);
    };
  };

  useEffect(() => {
    // if (mydata.nick != "-") setIsMember(true);
    if (localStorage.getItem("mydata"))
      setMyData(JSON.parse(localStorage.getItem("mydata")));
  }, []);

  useEffect(() => {
    setRankingpopup(false);
    setMypagepopup(false);
    setIsBgm(true);
  }, []);

  return (
    // <div style={mainstyle}>
    <div className={mainstyleClass}>
      {noticedata.length > 0 ? (
        <div className={noticeareastyleClass}>
          {noticedata.map((index) => {
            return <Noticebar key={index} />;
          })}
        </div>
      ) : null}
      <Menubar
        data={{ image: mydata.image, nickname: mydata.nick, rank: mydata.rank }}
      />
      {/* <div style={rankingpanelstyle}> */}
      <div
        // className={rankpopup}
        className={rankingpanelstyleClass}
      >
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
          setImgStyle(dialogtextstyleClass + " bg-npcLeft bg-cover");
          NPCClick();
        }}
      ></div>
      {/* <div style={maleNPC} onClick={NPCClick}></div> */}
      <div
        className={maleNPCClass}
        onClick={() => {
          setImgStyle(dialogtextstyleClass + " bg-npcRight bg-cover");
          NPCClick();
        }}
      ></div>
      {/* <div style={textStyle}>{text}</div> */}
      <div className={textStyle}>
        <div className={imgStyle}>{text}</div>
      </div>
      <Buttonbar />
    </div>
  );
}

export default Main;
