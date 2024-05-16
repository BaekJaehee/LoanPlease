import axios from "axios";
import { Cookies } from "react-cookie";

const cookie = new Cookies();
const token = cookie.get("Authorization");
axios.defaults.headers.Authorization = `Bearer ${token}`;

export const tokenrefresh = async () => {
  const url = "/api/refresh";
  return await axios
    .get(url, {
      withCredentials: true,
    })
    .then((response) => {
      if (response.status == 200) return response.data;
      else console.log(response);
    })
    .catch((e) => console.log(e));
};

export const getUserInfo = async () => {
  const url = "/api/user-info";
  return await axios
    .get(url, {
      withCredentials: true,
    })
    .then((response) => {
      if (response.data.status == 200) return response.data.dataBody;
      else console.log(response);
    })
    .catch((e) => {
      console.log(e);
    });
};

export const nicknameCheck = async (nickname) => {
  const url = `/api/auth/nickname/${nickname}`;
  return await axios
    .get(url, {
      withCredentials: true,
    })
    .then((response) => {
      if (response.status == 200) return true;
      else return false;
    })
    .catch((e) => {
      console.log(e);
    });
};

export const signup = async (data) => {
  const url = "/api/auth/register";
  return await axios
    .post(url, data, {
      withCredentials: true,
    })
    .then((response) => {
      if (response.data.status == 200) return response.data.dataBody;
      else console.log(response);
    })
    .catch((e) => {
      console.log(e);
    });
};

export const changeInfo = async (token, data) => {
  const url = "/api/auth";
  return await axios
    .put(url, data, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    .then((response) => {
      if (response.status == 200) return true;
      else return false;
    })
    .catch((e) => {
      console.log(e);
    });
};

export const exit = async (token, id) => {
  const url = `/api/auth/${id}`;
  return await axios
    .delete(url, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    .then((response) => {
      if (response.status == 200) return true;
      else return false;
    })
    .catch((e) => {
      console.log(e);
    });
};

export const friendsearch = async () => {
  const url = "/api/friends";
  return await axios
    .get(url)
    .then((response) => {
      if (response.data.status == 200) {
        return response.data.dataBody.friends;
      } else console.log(response);
    })
    .catch((e) => console.log(e));
};

export const friendsearchByname = async (input) => {
  const url = `/api/friends/search/${input}`;
  return await axios
    .get(url)
    .then((response) => {
      if (response.data.data) {
        return response.data.data.friends;
      } else console.log(response);
    })
    .catch((e) => console.log(e));
};

export const friendrequest = async (id) => {
  const url = "/api/friends/request";
  return await axios
    .post(url, {})
    .then((response) => {})
    .catch((e) => console.log(e));
};

export const friendresponse = async (id, answer) => {
  const url = "/api/friends/response";
  return await axios
    .post(url, {})
    .then((response) => {})
    .catch((e) => console.log(e));
};
