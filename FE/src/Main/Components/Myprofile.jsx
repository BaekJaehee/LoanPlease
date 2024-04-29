import myavatar from "./assets/myavatar.png";
import myicon from "./assets/myicon.png";

const myprofilestyleClass = `
flex justify-around items-center 
w-full h-[35%] 
`;

const myavatarareaClass = `
flex flex-col justify-around items-center 
w-[20%] h-full ml-8
`;

const avatarbuttonstyleClass = `
flex justify-center items-center 
w-36 h-12 rounded-[20px] mx-5 my-2.5 
font-cusFont2 text-base text-white cursor-pointer 
bg-gradient-to-l from-[#67c6e3] to-[#378ce7] 
`;

const myinfoareaClass = `
flex flex-col justify-center items-center 
w-[60%] h-full 
`;

const linestyleClass = `
flex flex-row justify-between items-baseline 
w-[60%] my-2.5 
`;

// const myprofilestyle = {
//   width: "100%",
//   height: "35%",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const myavatararea = {
//   width: "20%",
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const avatarbuttonstyle = {
//   width: "144px",
//   height: "48px",
//   borderRadius: "20px",
//   margin: "10px 20px",
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
//   fontFamily: "Orbit",
//   fontSize: "16px",
//   color: "#ffffff",
//   backgroundImage:
//     "linear-gradient(180deg, rgb(103, 198, 227) 0%, rgb(55, 140, 231) 100%)",
// };

// const myinfoarea = {
//   width: "60%",
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "center",
//   alignItems: "center",
// };

// const linestyle = {
//   width: "60%",
//   margin: "10px 0px",
//   display: "flex",
//   flexDirection: "row",
//   justifyContent: "space-between",
//   alignItems: "baseline",
// };

function Myprofile() {
  return (
    // <div style={myprofilestyle}>
    <div className={myprofilestyleClass}>
      {/* <div style={myavatararea}> */}
      <div className={myavatarareaClass}>
        <div>
          <img width={60} src={myavatar} />
        </div>
        {/* <div style={avatarbuttonstyle}>아바타 바꾸기</div> */}
        <div className={avatarbuttonstyleClass}>아바타 바꾸기</div>
      </div>
      {/* <div style={myinfoarea}> */}
      <div className={myinfoareaClass}>
        {/* <div style={linestyle}> */}
        <div className={linestyleClass}>
          {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "16px" }}> */}
          <div className="font-cusFont1 text-base">닉네임</div>
          {/* <div style={{ fontFamily: "Orbit", fontSize: "12px" }}> */}
          <div className="font-cusFont2 text-xs">loan_please</div>
          {/* <div
            style={{ fontFamily: "Orbit", fontSize: "8px", cursor: "pointer" }}
          > */}
          <div className="cursor-pointer font-cusFont2 text-[8px]">
            변경하기
          </div>
        </div>
        {/* <div style={linestyle}> */}
        <div className={linestyleClass}>
          {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "16px" }}> */}
          <div className="font-cusFont1 text-base">이메일</div>
          {/* <div style={{ width: "60%", fontFamily: "Orbit", fontSize: "12px" }}> */}
          <div className="w-[60%] font-cusFont2 text-xs">ssafy@gmail.com</div>
        </div>
      </div>
      <img width={70} src={myicon} />
    </div>
  );
}

export default Myprofile;
