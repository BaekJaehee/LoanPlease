import { useEffect } from "react";
import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const setMyData = LoginStore((state) => state.setMyData);

  useEffect(() => {
    if (localStorage.getItem("mydata")) {
      setIsLogin(true);
      setMyData(JSON.parse(localStorage.getItem("mydata")));
    }
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
