import Login from "./Components/Login";
import Main from "./Components/Mainpage";
import { LoginStore } from "./Store";

function Home() {
  const isLogin = LoginStore((state) => state.isLogin);

  return <>{isLogin ? <Main /> : <Login />}</>;
}

export default Home;
