import axios from "axios";

const isNewMember = async (email) => {
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

const nicknameCheck = async (nickname) => {
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

const signup = async (data) => {
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

const changeInfo = async (token, data) => {
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

const exit = async (token, id) => {
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
