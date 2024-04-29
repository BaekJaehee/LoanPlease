import Myprofile from "./Myprofile";
import Mycredit from "./Mycredit";

const mypagestyleClass = `
w-[80%] h-full 
bg-white px-[50px] py-[30px]  
border-[20px] border-[#FFC436] rounded-[40px] 
`;

const mypagetitlestyleClass = `
w-full font-cusFont1 text-[50px] text-center 
`;

// const mypagestyle = {
//   zIndex: "2",
//   width: "90%",
//   height: "100%",
//   backgroundColor: "#ffffff",
//   border: "20px solid #FFC436",
//   borderRadius: "40px",
//   padding: "20px 50px",
//   transform: "translateX(5%) translateY(3%)",
// };

// const mypagetitlestyle = {
//   fontFamily: "비트비트체v2",
//   fontSize: "50px",
//   width: "100%",
//   textAlign: "center",
// };

function Mypage() {
  return (
    // <div style={mypagestyle}>
    <div className={mypagestyleClass}>
      {/* <div style={mypagetitlestyle}>Profile / Setting</div> */}
      <div className={mypagetitlestyleClass}>Profile / Setting</div>
      <Myprofile />
      <Mycredit />
    </div>
  );
}

export default Mypage;
