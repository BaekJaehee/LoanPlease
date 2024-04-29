/* eslint-disable react/prop-types */

const rankingdatastyleClass = `
flex justify-around items-center 
w-[262px] h-[55px] 
font-cusFont2 
`;

const rankingimageareaClass = `
w-[50px] h-[50px] 
bg-[#D9D9D9] rounded-[100px] object-cover 
`;

const rankinginfoareaClass = `
flex flex-col justify-around 
w-[200px] h-full 
`;

const rankstyleClass = "text-xs";

const rankinfostyleClass = `
flex justify-between items-center text-xl 
`;

// const rankingdatastyle = {
//   width: "262px",
//   height: "55px",
//   fontFamily: "Orbit",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const rankingimagearea = {
//   width: "50px",
//   height: "50px",
//   backgroundColor: "#D9D9D9",
//   borderRadius: "100px",
//   objectFit: "cover",
// };

// const rankinginfoarea = {
//   width: "200px",
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "space-around",
// };

// const rankstyle = {
//   fontSize: "12px",
// };

// const rankinfostyle = {
//   display: "flex",
//   justifyContent: "space-between",
//   alignItems: "center",
//   fontSize: "20px",
// };

function Rankingdata(props) {
  return (
    // <div style={rankingdatastyle}>
    <div className={rankingdatastyleClass}>
      {/* <div style={rankingimagearea}></div> */}
      <div className={rankingimageareaClass}>
        <img src={props.data.image} />
      </div>
      {/* <div style={rankinginfoarea}> */}
      <div className={rankinginfoareaClass}>
        {/* <span style={rankstyle}>{props.rank}위</span> */}
        <span className={rankstyleClass}>{props.rank}위</span>
        {/* <div style={rankinfostyle}> */}
        <div className={rankinfostyleClass}>
          <div>{props.data.nickname}</div>
          <div>{props.data.score}</div>
        </div>
      </div>
    </div>
  );
}

export default Rankingdata;
