import rankingbutton from "./assets/rankingbutton.png";
import startbutton from "./assets/startbutton.png";
import mypagebutton from "./assets/mypagebutton.png";

const buttonareastyle = {
  width: "100%",
  padding: "24px",
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-around",
  alignItems: "baseline",
};

function Buttonbar() {
  return (
    <div style={buttonareastyle}>
      <img
        width="30%"
        height={90}
        style={{ maxWidth: "300px", cursor: "pointer" }}
        onClick={() => {}}
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
        onClick={() => {}}
        src={mypagebutton}
      />
    </div>
  );
}

export default Buttonbar;
