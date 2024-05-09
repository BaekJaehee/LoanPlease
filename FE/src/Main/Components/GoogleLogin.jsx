import { useEffect } from "react";
import { LoginStore } from "../Store";
import { useNavigate } from "react-router-dom";
import Signup from "./Signup";

function GoogleLogin() {
  const ismember = LoginStore((state) => state.ismember);
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const navigate = useNavigate();

  useEffect(() => {
    if (ismember) {
      setIsLogin(true);
      navigate("/");
    }
  }, []);

  return <div>{!ismember ? <Signup /> : null}</div>;
}

export default GoogleLogin;
