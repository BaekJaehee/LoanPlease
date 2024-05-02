import { LoginStore } from "../Store";

const mycreditstyleClass = `
flex justify-around items-center 
w-full h-[35%] 
`;

const creditareaClass = `
flex flex-col justify-around items-center h-full 
`;

const resultareaClass = `
flex justify-around items-center 
w-[50%] h-full bg-[#FFD28F] 
`;

const resultscorestyleClass = `
flex justify-center items-center 
w-20 h-20 rounded-[160px] 
`;

const buttonareaClass = `
flex flex-col justify-center items-center h-full 
`;

// const mycreditstyle = {
//   width: "100%",
//   height: "35%",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const creditarea = {
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const resultarea = {
//   width: "50%",
//   height: "100%",
//   backgroundColor: "#FFD28F",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const resultscorestyle = {
//   zIndex: "5",
//   width: "80px",
//   height: "80px",
//   borderRadius: "160px",
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
// };

// const buttonarea = {
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "center",
//   alignItems: "center",
// };

function Mycredit() {
  const setIsLogin = LoginStore((state) => state.setIsLogin);

  return (
    // <div style={mycreditstyle}>
    <div className={mycreditstyleClass}>
      {/* <div style={creditarea}> */}
      <div className={creditareaClass}>
        {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "14px" }}> */}
        <div className="mt-5 font-cusFont1 text-sm">나의 신용점수는?</div>
        <div
          onClick={() => {}}
          // style={{ fontFamily: "Orbit", fontSize: "12px", cursor: "pointer" }}
          className="mb-5 cursor-pointer font-cusFont2 text-xs"
        >
          서류 제출하고 점수 확인하기
        </div>
      </div>
      {/* <div style={resultarea}> */}
      <div className={resultareaClass}>
        {/* <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        > */}
        <div className="flex flex-col items-center justify-center">
          {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "12px" }}> */}
          <div className="font-cusFont1 text-xs">NICE</div>
          {/* <div style={{ ...resultscorestyle, backgroundColor: "#5DEBD7" }}> */}
          <div className={resultscorestyleClass + "bg-[#5DEBD7]"}>815점</div>
        </div>
        {/* <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        > */}
        <div className="flex flex-col items-center justify-center">
          {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "12px" }}> */}
          <div className="font-cusFont1 text-xs">KCB</div>
          {/* <div style={{ ...resultscorestyle, backgroundColor: "#C5FF95" }}> */}
          <div className={resultscorestyleClass + "bg-[#C5FF95]"}>760점</div>
        </div>
      </div>
      {/* <div style={buttonarea}> */}
      <div className={buttonareaClass}>
        <div
          onClick={() => {
            setIsLogin(false);
          }}
          // style={{
          //   fontFamily: "비트비트체v2",
          //   fontSize: "16px",
          //   cursor: "pointer",
          //   marginBottom: "10px",
          // }}
          className="mb-2.5 cursor-pointer font-cusFont1 text-base"
        >
          Logout
        </div>
        <div
          onClick={() => {
            setIsLogin(false);
          }}
          // style={{
          //   fontFamily: "Orbit",
          //   fontSize: "12px",
          //   color: "#A9A9A9",
          //   cursor: "pointer",
          //   marginTop: "10px",
          // }}
          className="mt-2.5 cursor-pointer font-cusFont2 text-xs text-[#A9A9A9]"
        >
          탈퇴하기
        </div>
      </div>
    </div>
  );
}

export default Mycredit;
