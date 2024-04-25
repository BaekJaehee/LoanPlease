import { useEffect, useState } from "react";
import { LoginStore } from "../Store";
import loginbutton from "./assets/loginbutton.png";
import background from "./assets/splash_final.jpg";

const mainarea = {
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-around",
  alignItems: "center",
  width: "80%",
  height: "80%",
  backgroundColor: "rgba(255, 255, 255, 0.64)",
  opacity: "0",
};

const mainstyle = {
  zIndex: "-1",
  position: "absolute",
  width: "100%",
  height: "100vh",
  filter: "blur(0px)",
};

const titlestyle = {
  fontFamily: "비트비트체v2",
  fontSize: "80px",
  color: "#186F65",
};

function Login() {
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const [bgstyle, setBgStyle] = useState(mainstyle);
  const [panelstyle, setPanelStyle] = useState(mainarea);

  useEffect(() => {
    const bgtimer = setTimeout(() => {
      setBgStyle({ ...bgstyle, filter: "blur(2px)", transition: "all 10s" });
    }, 1500);
    const paneltimer = setTimeout(() => {
      setPanelStyle({ ...panelstyle, opacity: "1", transition: "all 3s" });
    }, 2000);
    return () => {
      clearTimeout(bgtimer);
      clearTimeout(paneltimer);
    };
  }, []);

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        height: "100vh",
      }}
    >
      <img style={bgstyle} src={background} />
      <div style={panelstyle}>
        <div style={titlestyle}>론플리즈</div>
        <img width={200} src="/loanplease.png" />
        <img
          onClick={() => {
            setIsLogin(true);
          }}
          width={300}
          style={{ cursor: "pointer" }}
          src={loginbutton}
        />
      </div>
    </div>
  );
}

export default Login;
