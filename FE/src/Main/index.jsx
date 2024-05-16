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

  // 토큰 기반 유저 정보 가져오는 함수. 토큰 만료 시 refresh 후 다시 정보 요청 시도함.
  const getmyinfo = async () => {
    getUserInfo()
      .then((value) => {
        if (value) {
          localStorage.setItem("mydata", JSON.stringify(value));
          setMyData(value);
          setIsLogin(true);
        } else {
          tokenrefresh
            .then((response) => {
              if (response) {
                getUserInfo()
                  .then((info) => {
                    localStorage.setItem("mydata", JSON.stringify(info));
                    setMyData(info);
                    setIsLogin(true);
                  })
                  .catch((e) => console.log(e));
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
        getmyinfo();
      }
    }
  }, []);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
