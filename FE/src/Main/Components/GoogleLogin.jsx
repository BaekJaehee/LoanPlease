import Signup from "./Signup";
import { Cookies } from "react-cookie";
import { LoginStore } from "./Store";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

function GoogleLogin() {
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const navigate = useNavigate();

  useEffect(() => {
    const cookie = new Cookies();
    const token = cookie.get("Authorization");
    if (token) {
      setIsLogin(true);
      navigate("/");
    }
  }, []);

  return (
    <div>
      <Signup />
    </div>
  );
}

export default GoogleLogin;
