// 로그인 후 리다이렉션 페이지. 회원가입 된 경우 자동 패스함
import { useEffect, useState } from "react";
import image from "./assets/myavatar.png";
import { LoginStore } from "../Store";
import { nicknameCheck, signup } from "../../API/API";
import { useNavigate } from "react-router-dom";
import { Cookies } from "react-cookie";

// const signupareastyle = {
//   zIndex: "10",
//   position: "absolute",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "start",
//   flexDirection: "column",
//   backgroundColor: "#ffffff",
//   width: "70%",
//   height: "70%",
//   margin: "10% 15%",
//   padding: "60px",
//   filter: "drop-shadow(0 0 9999px #000000)",
// };

// const contentareastyle = {
//   display: "flex",
//   justifyContent: "center",
//   flexDirection: "column",
//   alignItems: "center",
//   width: "50%",
//   height: "80%",
//   transform: "translateX(50%)",
// };

// const inputareastyle = {
//   width: "100%",
//   maxWidth: "200px",
//   outline: "none",
//   fontFamily: "Orbit",
//   fontSize: "16px",
//   borderBottom: "2px solid",
//   margin: "0px 10px",
// };

// const signupbuttonstyle = {
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
//   width: "50%",
//   height: "80px",
//   margin: "0% 25%",
//   backgroundColor: "#795458",
//   fontFamily: "비트비트체v2",
//   fontSize: "24px",
//   color: "#ffffff",
//   cursor: "pointer",
// };

const signupareastyleClass = `
z-10 absolute flex flex-col justify-around items-start 
bg-white w-[70%] h-full mx-[15%] p-[60px] 
translate-x-[-150%]
`;

const contentareastyleClass = `
flex flex-col justify-around items-center 
min-w-[150px] w-full h-[80%] my-5  
`;

const inputareastyleClass = `
w-full max-w-[350px] 
outline-none border-b-2 
font-cusFont2 mx-2.5 
`;

const validstyleClass = `
flex justify-center items-center 
w-20 h-10 rounded-[20px] bg-cusColor1 
text-white font-cusFont1 text-xs cursor-pointer
`;

const signupbuttonstyleClass = `
flex justify-center items-center 
w-1/2 h-[80px] mx-[25%] 
bg-cusColor2 font-cusFont1 text-white text-2xl cursor-pointer 
`;

function Signup() {
  const [img, setImg] = useState(`${image}`);
  const [nickname, setNickname] = useState("");
  const [email, setEmail] = useState("");

  const [mainstyle, setMainStyle] = useState(signupareastyleClass);
  const [check, setCheck] = useState(false);
  const [valid, setValid] = useState(false);
  const [comment, setComment] = useState("");

  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const setMyData = LoginStore((state) => state.setMyData);

  const navigate = useNavigate();
  const cookie = new Cookies();

  useEffect(() => {
    setEmail(cookie.get("tmpEmail"));
    setImg(cookie.get("tmpImage"));
    setTimeout(() => {
      setMainStyle(
        mainstyle.replace(
          "translate-x-[-150%]",
          "translate-x-0 transition-all duration-[1000ms]",
        ),
      );
    }, 1000);
  }, []);

  // const changeImg = (e) => {
  //   const file = e.target.files[0];
  //   if (file === null) return;
  //   const url = URL.createObjectURL(file);
  //   if (url !== "") setImg(url);
  // };

  const test = async () => {
    setCheck(true);
    if (nickname == "") setComment("닉네임을 입력해주세요");
    else {
      const result = await nicknameCheck(nickname);
      result ? setValid(true) : setValid(false);
      result
        ? setComment("사용 가능한 닉네임입니다")
        : setComment("닉네임이 중복됩니다");
    }
  };

  // 테스트용 입력 데이터
  // useEffect(() => {
  //   setNickname("loanplease");
  //   setEmail("ssafy@gmail.com");
  // }, []);

  const signupComplete = async () => {
    if (nickname == "") alert("닉네임은 필수 입력값입니다");
    if (!valid) alert("닉네임을 확인해주세요");
    // if (img != "") {
    //   const profileResult = await uploadimage(img);
    //   if (!profileResult)
    //     alert("이미지 업로드에 실패하였습니다. 다시 시도해주세요");
    // }
    else {
      const data = {
        nickname: nickname,
        profileImage: img,
        email: email,
      };
      const result = await signup(data);
      if (result) {
        setIsLogin(true);
        navigate("/");
        setMyData(result.dataBody);
        localStorage.setItem("mydata", JSON.stringify(result.dataBody));
      }
    }
  };

  return (
    // <div style={signupareastyle}>
    <div className={mainstyle}>
      <div>
        <p
          // style={{
          //   fontFamily: "비트비트체v2",
          //   fontSize: "36px",
          //   color: "#FFC94A",
          // }}
          className="translate-x-1/2 font-cusFont1 text-[36px] text-cusColor3"
        >
          은행원이 된 것을 환영합니다!
        </p>
        <p
          // style={{
          //   fontFamily: "비트비트체v2",
          //   fontSize: "20px",
          // }}
          className="translate-x-[60%] font-cusFont1 text-[20px]"
        >
          게임에서 사용할 계정 정보를 확인해주세요
        </p>
      </div>
      {/* <div style={contentareastyle}> */}
      <div className={contentareastyleClass}>
        <div>
          <div
            // style={{
            //   display: "flex",
            //   flexDirection: "column",
            //   height: "90px",
            //   justifyContent: "space-between",
            //   alignItems: "center",
            //   margin: "10px 0px",
            // }}
            className="my-2.5 flex h-[90px] flex-col items-center justify-between"
          >
            <img
              // style={{ maxWidth: "50px", maxHeight: "50px" }}
              className="max-h-[50px] max-w-[50px]"
              width={50}
              src={img}
            />
            {/* <label
              // style={{
              //   fontFamily: "Orbit",
              //   fontSize: "16px",
              //   cursor: "pointer",
              // }}
              className="cursor-pointer font-cusFont2 text-base"
              htmlFor="imageicon"
            >
              프로필 사진 바꾸기
            </label>
            <input
              type="file"
              id="imageicon"
              accept="image/*"
              onChange={(e) => {
                changeImg(e);
                e.target.value = "";
              }}
              className="hidden"
            /> */}
          </div>
        </div>
        <div className="flex w-[60%] min-w-[400px]">
          {/* <span style={{ fontFamily: "비트비트체v2", fontSize: "24px" }}> */}
          <span className="font-cusFont1 text-2xl">닉네임 :</span>
          <input
            // style={inputareastyle}
            className={inputareastyleClass}
            value={nickname}
            onChange={(e) => {
              setNickname(e.target.value);
            }}
            placeholder="당신의 멋진 닉네임을 입력해주세요!"
          />
          <div
            onClick={async () => {
              test();
            }}
            className={validstyleClass}
          >
            중복 확인
          </div>
        </div>
        <div
          className={
            check
              ? "block w-[50%] " + (valid ? "text-green-600" : "text-red-600")
              : "hidden"
          }
        >
          {comment}
        </div>
        <div className="flex w-[60%] min-w-[300px]">
          {/* <span style={{ fontFamily: "비트비트체v2", fontSize: "24px" }}> */}
          <span className="font-cusFont1 text-2xl">이메일 :</span>
          {/* <input style={inputareastyle} value={email} /> */}
          <input
            className={inputareastyleClass}
            value={email}
            disabled={true}
          />
        </div>
      </div>
      <div
        onClick={async () => {
          signupComplete();
        }}
        // style={signupbuttonstyle}
        className={signupbuttonstyleClass}
      >
        출근 준비 완료!
      </div>
    </div>
  );
}

export default Signup;
