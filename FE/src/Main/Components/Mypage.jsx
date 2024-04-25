import Myprofile from "./Myprofile";
import Mycredit from "./Mycredit";

const mypagestyle = {
  zIndex: "2",
  width: "90%",
  height: "100%",
  backgroundColor: "#ffffff",
  border: "20px solid #FFC436",
  borderRadius: "40px",
  padding: "20px 50px",
  transform: "translateX(5%) translateY(3%)",
};

const mypagetitlestyle = {
  fontFamily: "비트비트체v2",
  fontSize: "50px",
  width: "100%",
  textAlign: "center",
};

function Mypage() {
  return (
    <div style={mypagestyle}>
      <div style={mypagetitlestyle}>Profile / Setting</div>
      <Myprofile />
      <Mycredit />
    </div>
  );
}

export default Mypage;
