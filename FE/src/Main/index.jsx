import { useEffect } from "react";
import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";
import { Cookies } from "react-cookie";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);
  const setIsLogin = LoginStore((state) => state.setIsLogin);

  useEffect(() => {
    const cookie = new Cookies();
    const token = cookie.get("Authorization");
    if (token) setIsLogin(true);
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
