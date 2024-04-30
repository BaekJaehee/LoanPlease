import rankingbutton from "./assets/rankingbutton.png";
import startbutton from "./assets/startbutton.png";
import mypagebutton from "./assets/mypagebutton.png";
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

const buttonareastyleClass = `
flex flex-row justify-around items-baseline 
w-full p-6 
`;

const buttonstyleClass = `
z-[1] w-[30%] max-w-[300px] h-[90px] cursor-pointer 
`;

function Buttonbar() {
  const navigate = useNavigate();

  const rankingpopup = MainStore((state) => state.rankingpopup);
  const setRankingpopup = MainStore((state) => state.setRankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const setMypagepopup = MainStore((state) => state.setMypagepopup);
  const setSearchmode = RankingStore((state) => state.setSearchmode);
  const setEditmode = MypageStore((state) => state.setEditmode);

  const makeClickSound = () => {
    const audio = new Audio("audioes/pop_sound.mp3");
    audio.play();
  };

  return (
    // <div style={buttonareastyle}>
    <div className={buttonareastyleClass}>
      <img
        // width="30%"
        // height={90}
        // style={{ maxWidth: "300px", cursor: "pointer" }}
        className={buttonstyleClass}
        onClick={() => {
          makeClickSound();
          setRankingpopup(!rankingpopup);
          setMypagepopup(false);
          setSearchmode(false);
          setEditmode(false);
        }}
        src={rankingbutton}
      />
      <img
        // width="30%"
        // height={120}
        // style={{ maxWidth: "300px", cursor: "pointer" }}
        className={buttonstyleClass.replace("h-[90px]", "h-[120px]")}
        onClick={() => {
          makeClickSound();
          navigate("/game");
        }}
        src={startbutton}
      />
      <img
        // width="30%"
        // height={90}
        // style={{ maxWidth: "300px", cursor: "pointer" }}
        className={buttonstyleClass}
        onClick={() => {
          makeClickSound();
          setMypagepopup(!mypagepopup);
          setRankingpopup(false);
          setSearchmode(false);
          setEditmode(false);
        }}
        src={mypagebutton}
      />
    </div>
  );
}

export default Buttonbar;
