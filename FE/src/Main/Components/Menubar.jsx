/* eslint-disable react/prop-types */
import shop from "./assets/shop.png";
import musicon from "./assets/bgm_on.png";
import musicoff from "./assets/bgm_off.png";
import { LoginStore, MainStore } from "../Store";
import { useNavigate } from "react-router-dom";

const menubarstyleClass = `
z-[1] flex flex-wrap justify-around items-center 
w-full bg-[#FFE3BB] 
`;

const menutitlestyleClass = `
min-w-[200px] font-cusFont1 text-[60px] 
`;

const profilestyleClass = `
flex justify-around items-center 
w-[35%] min-w-[250px] h-[100px] 
bg-[#0C356A] 
`;

const profileimagestyleClass = `
flex justify-center items-center 
w-[60px] min-w-[30px] h-[60px] 
rounded-full bg-[#D9D9D9] 
`;

// const menubarstyle = {
//   width: "100%",
//   display: "flex",
//   flexWrap: "wrap",
//   justifyContent: "space-around",
//   alignItems: "center",
//   backgroundColor: "#FFE3BB",
// };

// const menutitlestyle = {
//   minWidth: "200px",
//   fontFamily: "비트비트체v2",
//   fontSize: "60px",
// };

// const profilestyle = {
//   width: "35%",
//   height: "100px",
//   minWidth: "250px",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
//   backgroundColor: "#0C356A",
// };

// const profileimagestyle = {
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
//   width: "60px",
//   minWidth: "30px",
//   height: "60px",
//   borderRadius: "1000px",
//   backgroundColor: "#D9D9D9",
// };

function Menubar(props) {
  const isLogin = LoginStore((state) => state.isLogin);
  const isBgm = MainStore((state) => state.isBgm);
  const setIsBgm = MainStore((state) => state.setIsBgm);

  const navigate = useNavigate();

  const makeClickSound = () => {
    const audio = new Audio("audioes/pop_sound.mp3");
    audio.play();
  };

  return (
    // <div style={menubarstyle}>
    <div className={menubarstyleClass}>
      <img width={70} height={70} src="/loanplease.png" />
      {/* <div style={menutitlestyle}>론플리즈</div> */}
      <div className={menutitlestyleClass}>론플리즈</div>
      {/* <div style={profilestyle}> */}
      <div className={profilestyleClass}>
        {/* <img style={profileimagestyle} src={isLogin ? props.data.image : ""} /> */}
        <img
          className={profileimagestyleClass}
          src={isLogin ? props.data.image : ""}
        />
        {/* <div style={{ fontFamily: "Orbit", color: "white", fontSize: "40px" }}> */}
        <div className="font-cusFont2 text-[40px] text-white">
          {isLogin ? props.data.nickname : ""}
        </div>
        {/* <div style={{ fontFamily: "Orbit", color: "white", fontSize: "24px" }}> */}
        <div className="font-cusFont2 text-2xl text-white">
          {isLogin ? `${props.data.rank}위` : ""}
        </div>
      </div>
      <div
        onClick={() => {
          navigate("/shop");
        }}
      >
        <img width={100} height={50} src={shop} />
      </div>
      <div
        // style={{ margin: "10px", cursor: "pointer" }}
        className="m-2.5 cursor-pointer"
        onClick={() => {
          makeClickSound();
          setIsBgm(!isBgm);
        }}
      >
        {isBgm ? <img src={musicon} /> : <img src={musicoff} />}
        {isBgm ? (
          <audio
            src="audioes/intro_main_bgm.mp3"
            autoPlay={true}
            loop={true}
          ></audio>
        ) : null}
      </div>
    </div>
  );
}

export default Menubar;
