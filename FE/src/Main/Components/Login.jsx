import { useState } from "react";
import loginbutton from "./assets/loginbutton.png";
import background from "./assets/splash_final.jpg";
import { LoginStore } from "../Store";
import { Cookies } from "react-cookie";

// const mainarea = {
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "space-around",
//   alignItems: "center",
//   width: "80%",
//   height: "80%",
//   backgroundColor: "rgba(255, 255, 255, 0.64)",
//   opacity: "0",
// };

// const mainstyle = {
//   zIndex: "-1",
//   position: "absolute",
//   width: "100%",
//   height: "100vh",
//   filter: "blur(0px)",
// };

// const titlestyle = {
//   fontFamily: "비트비트체v2",
//   fontSize: "80px",
//   color: "#186F65",
// };

const loginstyleClass = `
flex justify-center items-center w-full h-screen
`;

const mainareaClass = `
flex flex-col 
justify-around items-center 
w-[80%] h-[80%]
bg-[#ffffffad] 
opacity-0
`;

const mainstyleClass = `
absolute z-[-1]  
w-full h-screen 
blur-none 
`;

const titlestyleClass = `
font-cusFont1 text-cusColor1 text-[80px]
`;

const startButtonClass = `
absolute z-[20] cursor-pointer 
font-cusFont1 text-[40px] text-cusColor5 opacity-100
`;

function Login() {
  // const [bgstyle, setBgStyle] = useState(mainstyle);
  const [bgstyle, setBgStyle] = useState(mainstyleClass);
  // const [panelstyle, setPanelStyle] = useState(mainarea);
  const [panelstyle, setPanelStyle] = useState(mainareaClass);

  const [startButton, setStartButton] = useState(startButtonClass);
  const [audio, setAudio] = useState(<div></div>);

  const setIsLogin = LoginStore((state) => state.setIsLogin);

  const init = () => {
    setStartButton(startButton.replace("opacity-100", "opacity-0"));

    setAudio(
      <iframe
        id="bgm"
        src="audioes/intro_bgm.mp3"
        allow="autoplay"
        style={{ display: "none" }}
      ></iframe>,
    );

    const bgtimer = setTimeout(() => {
      // setBgStyle({ ...bgstyle, filter: "blur(2px)", transition: "all 10s" });
      setBgStyle(
        bgstyle.replace("blur-none", "blur-[2px]") +
          " transition-all duration-[5000ms]",
      );
    }, 1500);
    const paneltimer = setTimeout(() => {
      // setPanelStyle({ ...panelstyle, opacity: "1", transition: "all 3s" });
      setPanelStyle(
        panelstyle.replace("opacity-0", "opacity-100") +
          " transition-all duration-[3000ms]",
      );
    }, 5000);
    return () => {
      clearTimeout(bgtimer);
      clearTimeout(paneltimer);
    };
  };

  return (
    // <div
    //   style={{
    //     display: "flex",
    //     justifyContent: "center",
    //     alignItems: "center",
    //     width: "100%",
    //     height: "100vh",
    //   }}
    // >
    <div className={loginstyleClass}>
      {audio}
      <div onClick={init} className={startButton}>
        Press To Start
      </div>
      {/* <iframe
        src="audioes/intro_bgm.mp3"
        allow="autoplay"
        id="audio"
        style={{ display: "none" }}
      /> */}
      {/* <img style={bgstyle} src={background} /> */}
      <img className={bgstyle} src={background} />
      {/* <div style={panelstyle}> */}
      <div className={panelstyle}>
        {/* <div style={titlestyle}>론플리즈</div> */}
        {/* <div style={titlestyleClass}>론플리즈</div> */}
        <div className={titlestyleClass}>론플리즈</div>
        <img width={200} src="/loanplease.png" />
        <img
          onClick={() => {
            // setIsLogin(true);
            // 구글 로그인 후 쿠키에 토큰 값 저장
            const url =
              "http://k10d105.p.ssafy.io:8082/oauth2/authorization/google";
            location.href = url;
            const cookie = new Cookies();
            const token = cookie.get("Authorization");
            if (token) {
              setIsLogin(true);
            }
          }}
          width={300}
          // style={{ cursor: "pointer" }}
          className="cursor-pointer"
          src={loginbutton}
        />
      </div>
    </div>
  );
}

export default Login;
