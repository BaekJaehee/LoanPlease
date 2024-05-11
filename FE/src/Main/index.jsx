import { useEffect } from "react";
import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";
import { Cookies } from 'react-cookie';
import { getUserInfo } from "../API/API";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);
  const setIsLogin = LoginStore((state) => state.setIsLogin);

  const myinfo = async (token) => {
    const cookie = new Cookies()
    const token = cookie.get('Authorization')
    const info =  await getUserInfo(token)
    if (info) {
      localStorage.setItem('mydata', JSON.stringify(info))
      setIsLogin(true)
    }
    else {
      localStorage.clear()
      setIsLogin(false)
    }
  }

  useEffect(() => {
    myinfo()
  }, [])

  useEffect(() => {
    if (localStorage.getItem("mydata")) setIsLogin(true);
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
