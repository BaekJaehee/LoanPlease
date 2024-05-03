import { useEffect } from "react";
import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);
  const setIsLogin = LoginStore((state) => state.setIsLogin);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) setIsLogin(true);
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
