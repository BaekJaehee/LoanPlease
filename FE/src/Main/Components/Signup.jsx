// 로그인 후 리다이렉션 페이지. 회원가입 된 경우 자동 패스함
import { useEffect, useState } from "react";
import image from "./assets/myavatar.png";
import { LoginStore } from "../Store";

const signupareastyle = {
  zIndex: "10",
  position: "absolute",
  display: "flex",
  justifyContent: "space-around",
  alignItems: "start",
  flexDirection: "column",
  backgroundColor: "#ffffff",
  width: "70%",
  height: "70%",
  margin: "10% 15%",
  padding: "60px",
  filter: "drop-shadow(0 0 9999px #000000)",
};

const contentareastyle = {
  display: "flex",
  justifyContent: "center",
  flexDirection: "column",
  alignItems: "center",
  width: "50%",
  height: "80%",
  transform: "translateX(50%)",
};

const inputareastyle = {
  width: "100%",
  maxWidth: "200px",
  outline: "none",
  fontFamily: "Orbit",
  fontSize: "16px",
  borderBottom: "2px solid",
  margin: "0px 10px",
};

const signupbuttonstyle = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  width: "50%",
  height: "80px",
  margin: "0% 25%",
  backgroundColor: "#795458",
  fontFamily: "비트비트체v2",
  fontSize: "24px",
  color: "#ffffff",
  cursor: "pointer",
};

function Signup() {
  const [img, setImg] = useState(`${image}`);
  const [nickname, setNickname] = useState("");
  const [email, setEmail] = useState("");

  const setIsMember = LoginStore((state) => state.setIsMember);
  const setMyData = LoginStore((state) => state.setMyData);

  useEffect(() => {
    setNickname("loanplease");
    setEmail("ssafy@gmail.com");
  }, []);

  const changeImg = (e) => {
    const file = e.target.files[0];
    if (file === null) return;
    const url = URL.createObjectURL(file);
    if (url !== "") setImg(url);
  };

  return (
    <div style={signupareastyle}>
      <div>
        <p
          style={{
            fontFamily: "비트비트체v2",
            fontSize: "36px",
            color: "#FFC94A",
          }}
        >
          은행원이 된 것을 환영합니다!
        </p>
        <p
          style={{
            fontFamily: "비트비트체v2",
            fontSize: "20px",
          }}
        >
          게임에서 사용할 계정 정보를 확인해주세요
        </p>
      </div>
      <div style={contentareastyle}>
        <div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              height: "90px",
              justifyContent: "space-between",
              alignItems: "center",
              margin: "10px 0px",
            }}
          >
            <img
              style={{ maxWidth: "50px", maxHeight: "50px" }}
              width={50}
              src={img}
            />
            <label
              style={{
                fontFamily: "Orbit",
                fontSize: "16px",
                cursor: "pointer",
              }}
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
            />
          </div>
        </div>
        <div>
          <span style={{ fontFamily: "비트비트체v2", fontSize: "24px" }}>
            닉네임 :
          </span>
          <input
            style={inputareastyle}
            value={nickname}
            onChange={(e) => {
              setNickname(e.target.value);
            }}
          />
        </div>
        <div>
          <span style={{ fontFamily: "비트비트체v2", fontSize: "24px" }}>
            이메일 :
          </span>
          <input style={inputareastyle} value={email} />
        </div>
      </div>
      <div
        onClick={() => {
          setIsMember(true);
          setMyData({ image: img, nick: nickname, address: email, rank: "-" });
        }}
        style={signupbuttonstyle}
      >
        출근 준비 완료!
      </div>
    </div>
  );
}

export default Signup;
