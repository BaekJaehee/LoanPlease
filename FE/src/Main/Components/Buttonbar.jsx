import rankingbutton from "./assets/rankingbutton.png";
import startbutton from "./assets/startbutton.png";
import mypagebutton from "./assets/mypagebutton.png";
import { MainStore, RankingStore } from "../Store";

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
  const rankingpopup = MainStore((state) => state.rankingpopup);
  const setRankingpopup = MainStore((state) => state.setRankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const setMypagepopup = MainStore((state) => state.setMypagepopup);
  const setSearchmode = RankingStore((state) => state.setSearchmode);

  return (
    // <div style={buttonareastyle}>
    <div className={buttonareastyleClass}>
      <img
        // width="30%"
        // height={90}
        // style={{ maxWidth: "300px", cursor: "pointer" }}
        className={buttonstyleClass}
        onClick={() => {
          setRankingpopup(!rankingpopup);
          setMypagepopup(false);
          setSearchmode(false);
        }}
        src={rankingbutton}
      />
      <img
        // width="30%"
        // height={120}
        // style={{ maxWidth: "300px", cursor: "pointer" }}
        className={buttonstyleClass.replace("h-[90px]", "h-[120px]")}
        onClick={() => {}}
        src={startbutton}
      />
      <img
        // width="30%"
        // height={90}
        // style={{ maxWidth: "300px", cursor: "pointer" }}
        className={buttonstyleClass}
        onClick={() => {
          setMypagepopup(!mypagepopup);
          setRankingpopup(false);
          setSearchmode(false);
        }}
        src={mypagebutton}
      />
    </div>
  );
}

export default Buttonbar;
