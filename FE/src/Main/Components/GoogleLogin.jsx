import { useEffect } from "react";
import { LoginStore } from "../Store";
import { useNavigate } from "react-router-dom";
import Signup from "./Signup";
import { Cookies } from "react-cookie";

function GoogleLogin() {
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const navigate = useNavigate();

  const cookie = new Cookies();

  return (
    <div>
      <Signup />
    </div>
  );
}

export default GoogleLogin;
