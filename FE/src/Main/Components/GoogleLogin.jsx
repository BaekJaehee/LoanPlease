import { useEffect } from "react";
import { Cookies } from "react-cookie";
import { LoginStore } from "../Store";
import { useNavigate } from "react-router-dom";

function GoogleLogin() {
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const navigate = useNavigate();

  useEffect(() => {
    // 구글 로그인 후 쿠키에 토큰 값 저장
    let url = "https://loanplease.kr/oauth2/authorization/google";
    location.href = url;
    const cookie = new Cookies();
    const token = cookie.get("Authorization");
    console.log(token);
    if (token) {
      localStorage.setItem("token", token);
      navigate("/");
      setIsLogin(true);
    }
  }, []);

  return <div></div>;
}

export default GoogleLogin;
