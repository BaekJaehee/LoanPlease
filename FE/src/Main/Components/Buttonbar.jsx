import rankingbutton from "./assets/rankingbutton.png";
import startbutton from "./assets/startbutton.png";
import mypagebutton from "./assets/mypagebutton.png";
import { MainStore, RankingStore } from "../Store";

const buttonareastyle = {
  width: "100%",
  padding: "24px",
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-around",
  alignItems: "baseline",
};

function Buttonbar() {
  const rankingpopup = MainStore((state) => state.rankingpopup);
  const setRankingpopup = MainStore((state) => state.setRankingpopup);
  const mypagepopup = MainStore((state) => state.mypagepopup);
  const setMypagepopup = MainStore((state) => state.setMypagepopup);
  const setSearchmode = RankingStore((state) => state.setSearchmode);

  return (
    <div style={buttonareastyle}>
      <img
        width="30%"
        height={90}
        style={{ maxWidth: "300px", cursor: "pointer" }}
        onClick={() => {
          setRankingpopup(!rankingpopup);
          setSearchmode(false);
        }}
        src={rankingbutton}
      />
      <img
        width="30%"
        height={120}
        style={{ maxWidth: "300px", cursor: "pointer" }}
        onClick={() => {}}
        src={startbutton}
      />
      <img
        width="30%"
        height={90}
        style={{ maxWidth: "300px", cursor: "pointer" }}
        onClick={() => {
          setMypagepopup(!mypagepopup);
          setSearchmode(false);
        }}
        src={mypagebutton}
      />
    </div>
  );
}

export default Buttonbar;
