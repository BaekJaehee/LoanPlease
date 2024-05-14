import { useEffect } from "react";
import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";
import { Cookies } from "react-cookie";
import { getUserInfo, tokenrefresh } from "../API/API";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);
  const setIsLogin = LoginStore((state) => state.setIsLogin);
  const setMyData = LoginStore((state) => state.setMyData);

  const getmyinfo = async () => {
    getUserInfo()
      .then((value) => {
        if (value) {
          localStorage.setItem("mydata", JSON.stringify(value));
          setMyData(value);
        } else {
          tokenrefresh()
            .then((response) => {
              if (response) {
                getUserInfo().then((result) => {
                  localStorage.setItem("mydata", JSON.stringify(result));
                  setMyData(result);
                });
              } else {
                localStorage.clear();
                setIsLogin(false);
              }
            })
            .catch((e) => console.log(e));
        }
      })
      .catch((e) => console.log(e));
  };

  useEffect(() => {
    if (localStorage.getItem("mydata")) {
      setIsLogin(true);
      setMyData(JSON.parse(localStorage.getItem("mydata")));
    } else {
      const cookie = new Cookies();
      const token = cookie.get("Authorization");
      if (token) {
        setIsLogin(true);
        // getmyinfo();
      }
    }
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
