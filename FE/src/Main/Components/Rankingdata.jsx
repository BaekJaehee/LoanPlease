/* eslint-disable react/prop-types */
const rankingdatastyle = {
  width: "262px",
  height: "55px",
  fontFamily: "Orbit",
  display: "flex",
  justifyContent: "space-around",
  alignItems: "center",
};

const rankingimagearea = {
  width: "50px",
  height: "50px",
  backgroundColor: "#D9D9D9",
  borderRadius: "100px",
  objectFit: "cover",
};

const rankinginfoarea = {
  width: "200px",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-around",
};

const rankstyle = {
  fontSize: "12px",
};

const rankinfostyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  fontSize: "20px",
};

function Rankingdata(props) {
  return (
    <div style={rankingdatastyle}>
      <div style={rankingimagearea}></div>
      <div style={rankinginfoarea}>
        <span style={rankstyle}>{props.rank}위</span>
        <div style={rankinfostyle}>
          <div>{props.data.nickname}</div>
          <div>{props.data.score}</div>
        </div>
      </div>
    </div>
  );
}

export default Rankingdata;
