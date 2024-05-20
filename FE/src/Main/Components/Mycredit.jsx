import { Cookies } from "react-cookie";
import { LoginStore } from "../Store";

const mycreditstyleClass = `
flex justify-around items-center 
w-full h-[35%] 
`;

const resultareaClass = `
flex flex-col justify-around items-start ps-5 
w-[80%] h-full bg-cusColor4 font-cusFont2 
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

      {/* <div style={resultarea}> */}
      <div className={resultareaClass}>
        <p>
          게임 내에서 활용하는 신용등급은 실제 등급 산출방식과 다소 차이가 있을
          수 있습니다.
        </p>
        <p>
          <span className="font-cusFont1 text-base text-cusColor2">
            {" '게임 시작하기' "}
          </span>
          버튼 위의 사람을 클릭하면 소소한 팁을 얻을 수 있습니다.
        </p>
        <p>
          상점에서 아이템을 구매하려면
          <span className="font-cusFont1 text-base text-cusColor3">
            {" '마이페이지' "}
          </span>
          버튼 위의 아바타를 클릭하세요
        </p>
      </div>
      {/* <div style={buttonarea}> */}
      <div className={buttonareaClass}>
        <div
          onClick={() => {
            localStorage.clear();
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
            if (window.confirm("정말 탈퇴하시겠습니까?")) {
              const cookie = new Cookies();
              if (cookie) {
                cookie.remove("Authorization");
                cookie.remove("RefreshToken");
              }
              localStorage.clear();
              setIsLogin(false);
            }
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
