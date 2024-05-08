import axios from "axios";

export const isNewMember = async (email) => {
  const url = `/api/email/${email}`;
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

export const nicknameCheck = async (nickname) => {
  const url = `/api/nickname/${nickname}`;
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
  const url = "/api/auth/info";
  return await axios
    .post(url, data)
    .then((response) => {
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
};

export const changeInfo = async (token, data) => {
  const url = "/api/auth/info";
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
