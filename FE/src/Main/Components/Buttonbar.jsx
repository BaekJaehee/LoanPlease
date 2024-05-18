import rankicon from "./assets/rank.png";
import starticon from "./assets/start.png";
import mypageicon from "./assets/mypage.png";
import { MainStore, MypageStore, RankingStore } from "../Store";
import { useNavigate } from "react-router-dom";

// const buttonareastyle = {
//   width: "100%",
//   padding: "24px",
//   display: "flex",
//   flexDirection: "row",
//   justifyContent: "space-around",
//   alignItems: "baseline",
// };

// const buttonareastyle = {
//   display: "flex",
//   flexDirection: "row",
//   justifyContent: "space-around",
//   alignItems: "end",
//   width: "100%",
//   padding: "24px",
// };

// const rankiconstyle = {
//   zIndex: "2",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
//   width: "30%",
//   maxWidth: "300px",
//   height: "90px",
//   padding: "0px 20px",
//   borderRadius: "60px",
//   backgroundColor: "#186F65",
//   fontFamily: "비트비트체v2",
//   fontSize: "40px",
//   color: "#ffffff",
//   cursor: "pointer",
// };

// const starticonstyle = {
//   zIndex: "2",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
//   flexDirection: "column",
//   width: "30%",
//   maxWidth: "300px",
//   height: "120px",
//   borderRadius: "60px",
//   border: "6px #795458 solid",
//   backgroundColor: "#FFFAB7",
//   fontFamily: "비트비트체v2",
//   fontSize: "30px",
//   color: "#795458",
//   cursor: "pointer",
// };

// const mypageiconstyle = {
//   zIndex: "2",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
//   width: "30%",
//   maxWidth: "300px",
//   height: "90px",
//   padding: "0px 20px",
//   borderRadius: "60px",
//   backgroundColor: "#5BBCFF",
//   fontFamily: "비트비트체v2",
//   fontSize: "40px",
//   color: "#FFC94A",
//   cursor: "pointer",
// };

const buttonareastyleClass = `
flex justify-around items-end w-full p-6
`;

const rankiconstyleClass = `
z-[2] flex justify-around items-center 
w-[30%] max-w-[300px] h-[90px] 
px-5 rounded-[40px] bg-cusColor1 
font-cusFont1 text-[40px] text-white cursor-pointer 
`;

const starticonstyleClass = `
z-[2] flex flex-col justify-around items-center 
w-[30%] max-w-[300px] h-[120px] 
rounded-[40px] border-[6px] border-cusColor2 
bg-cusColor4 font-cusFont1 text-[30px] text-cusColor2 cursor-pointer 
`;

const mypageiconstyleClass = `
z-[2] flex justify-around items-center 
w-[30%] max-w-[300px] h-[90px] px-5 
rounded-[40px] bg-cusColor5 
font-cusFont1 text-[40px] text-cusColor3 cursor-pointer 
`;

function Buttonbar() {
  const navigate = useNavigate();

  const rankingpopup = MainStore((state) => state.rankingpopup);
  const setRankingpopup = MainStore((state) => state.setRankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const setMypagepopup = MainStore((state) => state.setMypagepopup);
  const setSearchmode = RankingStore((state) => state.setSearchmode);
  const setEditmode = MypageStore((state) => state.setEditmode);
  const setIsBgm = MainStore((state) => state.setIsBgm)

  const makeClickSound = () => {
    const audio = new Audio("audioes/pop_sound.mp3");
    audio.play();
  };

  return (
    // <div style={buttonareastyle}>
    <div className={buttonareastyleClass}>
      <div
        onClick={() => {
          makeClickSound();
          setRankingpopup(!rankingpopup);
          setMypagepopup(false);
          setSearchmode(false);
          setEditmode(false);
        }}
        // style={rankiconstyle}
        className={rankiconstyleClass}
      >
        <img width={60} src={rankicon} />
        랭킹보기
      </div>
      <div
        onClick={() => {
          makeClickSound();
          setIsBgm(false);
          navigate("/game");
        }}
        // style={starticonstyle}
        className={starticonstyleClass}
      >
        <img width={40} src={starticon} />
        게임 시작하기
      </div>
      <div
        onClick={() => {
          makeClickSound();
          setMypagepopup(!mypagepopup);
          setRankingpopup(false);
          setSearchmode(false);
          setEditmode(false);
        }}
        // style={mypageiconstyle}
        className={mypageiconstyleClass}
      >
        <img width={50} src={mypageicon} />
        마이페이지
      </div>
    </div>
  );
}

export default Buttonbar;
