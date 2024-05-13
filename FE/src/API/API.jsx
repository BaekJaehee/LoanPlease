import axios from "axios";

export const getUserInfo = async () => {
  const url = "/api/user-info";
  return await axios
    .get(url)
    .then((response) => {
      if (response.status == 200) return response.data;
      else return {};
    })
    .catch((e) => {
      console.log(e);
    });
};

export const uploadimage = async (file) => {
  const url = "/api/upload";
  const form = new FormData();
  form.append("image", file);
  return await axios
    .post(url, form)
    .then((response) => {
      console.log(response.data);
      if (response.status == 200) return true;
      else return false;
    })
    .catch((e) => {
      console.log(e);
    });
};

export const nicknameCheck = async (nickname) => {
  const url = `/api/auth/nickname/${nickname}`;
  return await axios
    .get(url)
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
  const config = {
    "Content-Type": "application/json",
  };
  return await axios
    .post(url, data, config)
    .then((response) => {
      if (response.status == 200) return response;
      else return {};
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
