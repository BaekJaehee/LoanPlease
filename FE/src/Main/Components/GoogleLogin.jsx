import { useEffect } from "react";
import { LoginStore } from "../Store";
import { useNavigate } from "react-router-dom";
import { isNewMember } from "../../API/API";
import Signup from "./Signup";

function GoogleLogin() {
  const ismember = LoginStore((state) => state.ismember);
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const navigate = useNavigate();

  const checkNewmember = async (email) => {
    const result = await isNewMember(email);
    console.log(result);
  };

  useEffect(() => {
    // 구글 로그인 후 쿠키에 토큰 값 저장
    let url = "https://loanplease.kr/oauth2/authorization/google";
    location.href = url;
    if (ismember) {
      navigate("/");
      setIsLogin(true);
    } else {
      navigate("/signup");
    }
  }, []);

  return !ismember ? <Signup /> : <div></div>;
}

export default GoogleLogin;
