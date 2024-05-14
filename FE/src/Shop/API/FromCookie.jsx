import { useCookies } from "react-cookie";

function FromCookie() {
  const [cookies] = useCookies(['Authorization']);
  const token = cookies.Authorization;
  return token;
}

export default FromCookie;