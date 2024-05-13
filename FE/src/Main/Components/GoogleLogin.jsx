import { useEffect } from "react";
import { LoginStore } from "../Store";
import { useNavigate } from "react-router-dom";
import Signup from "./Signup";
import { Cookies } from "react-cookie";
import { getUserInfo } from "../../API/API";

function GoogleLogin() {
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const setMyData = LoginStore((state) => state.setMyData);
  const navigate = useNavigate();

  const login = async () => {
    const cookie = new Cookies();
    const token = cookie.get("Authorization");
    if (token) {
      const info = await getUserInfo(token);
      return info;
    }
  };

  useEffect(() => {
    login().then((value) => {
      if (value) {
        localStorage.setItem("mydata", JSON.stringify(value));
        setMyData(value);
        setIsLogin(true);
        navigate("/");
      }
    });
  }, []);

  return (
    <div>
      <Signup />
    </div>
  );
}

export default GoogleLogin;
