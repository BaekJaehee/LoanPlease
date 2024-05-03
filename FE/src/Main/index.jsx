import { useEffect } from "react";
import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";
import { Cookies } from "react-cookie";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);
  const setIsLogin = LoginStore((state) => state.setIsLogin);

  useEffect(() => {
    const cookies = new Cookies();
    const string = cookies.get("Authorization");
    console.log(string);
    if (string) setIsLogin(true);
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
