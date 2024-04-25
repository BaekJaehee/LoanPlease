/* eslint-disable react/prop-types */
import musicon from "./assets/bgm_on.png";
import musicoff from "./assets/bgm_off.png";
import { LoginStore, MainStore } from "../Store";

const menubarstyle = {
  width: "100%",
  display: "flex",
  flexWrap: "wrap",
  justifyContent: "space-around",
  alignItems: "center",
  backgroundColor: "#FFE3BB",
};

const menutitlestyle = {
  minWidth: "200px",
  fontFamily: "비트비트체v2",
  fontSize: "60px",
};

const profilestyle = {
  width: "35%",
  height: "100px",
  minWidth: "250px",
  display: "flex",
  justifyContent: "space-around",
  alignItems: "center",
  backgroundColor: "#0C356A",
};

const profileimagestyle = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  width: "60px",
  minWidth: "30px",
  height: "60px",
  borderRadius: "1000px",
  backgroundColor: "#D9D9D9",
};

function Menubar(props) {
  const isLogin = LoginStore((state) => state.isLogin);
  const isBgm = MainStore((state) => state.isBgm);
  const setIsBgm = MainStore((state) => state.setIsBgm);

  return (
    <div style={menubarstyle}>
      <img width={70} height={70} src="/loanplease.png" />
      <div style={menutitlestyle}>론플리즈</div>
      <div style={profilestyle}>
        <img style={profileimagestyle} src={isLogin ? props.data.image : ""} />
        <div style={{ fontFamily: "Orbit", color: "white", fontSize: "40px" }}>
          {isLogin ? props.data.nickname : ""}
        </div>
        <div style={{ fontFamily: "Orbit", color: "white", fontSize: "24px" }}>
          {isLogin ? `${props.data.rank}위` : ""}
        </div>
      </div>
      <div
        style={{ margin: "10px", cursor: "pointer" }}
        onClick={() => {
          setIsBgm(!isBgm);
        }}
      >
        {isBgm ? <img src={musicon} /> : <img src={musicoff} />}
      </div>
    </div>
  );
}

export default Menubar;
