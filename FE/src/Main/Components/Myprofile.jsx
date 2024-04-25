import myavatar from "./assets/myavatar.png";
import myicon from "./assets/myicon.png";

const myprofilestyle = {
  width: "100%",
  height: "35%",
  display: "flex",
  justifyContent: "space-around",
  alignItems: "center",
};

const myavatararea = {
  width: "20%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-around",
  alignItems: "center",
};

const avatarbuttonstyle = {
  width: "144px",
  height: "48px",
  borderRadius: "20px",
  margin: "10px 20px",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  fontFamily: "Orbit",
  fontSize: "16px",
  color: "#ffffff",
  backgroundImage:
    "linear-gradient(180deg, rgb(103, 198, 227) 0%, rgb(55, 140, 231) 100%)",
};

const myinfoarea = {
  width: "60%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
};

const linestyle = {
  width: "60%",
  margin: "10px 0px",
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "baseline",
};

function Myprofile() {
  return (
    <div style={myprofilestyle}>
      <div style={myavatararea}>
        <div>
          <img width={60} src={myavatar} />
        </div>
        <div style={avatarbuttonstyle}>아바타 바꾸기</div>
      </div>
      <div style={myinfoarea}>
        <div style={linestyle}>
          <div style={{ fontFamily: "비트비트체v2", fontSize: "16px" }}>
            닉네임
          </div>
          <div style={{ fontFamily: "Orbit", fontSize: "12px" }}>
            loan_please
          </div>
          <div
            style={{ fontFamily: "Orbit", fontSize: "8px", cursor: "pointer" }}
          >
            변경하기
          </div>
        </div>
        <div style={linestyle}>
          <div style={{ fontFamily: "비트비트체v2", fontSize: "16px" }}>
            이메일
          </div>
          <div style={{ width: "60%", fontFamily: "Orbit", fontSize: "12px" }}>
            ssafy@gmail.com
          </div>
        </div>
      </div>
      <img width={70} src={myicon} />
    </div>
  );
}

export default Myprofile;
