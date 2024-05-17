import axios from "axios";
import { Cookies } from "react-cookie";

// 192.168.100.178:8080
const API_BASE_URL = import.meta.env.VITE_BASE_URL;

const cookies = new Cookies();
const token = cookies.get('Authorization');

// 아이템 리스트 조회
export const itemsList = async () => {
  const url = `${API_BASE_URL}/api/store/items`;
  return await axios
    .get(url, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
  };

// 슬롯 구매
// 유저 슬롯개수 +1, 포인트 차감
export const purchaseSlot = async () => {
  const url = `${API_BASE_URL}/api/store/items/slot`;
  return await axios
    .post(url, {}, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      console.log(response);
      console.log(response.data);
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}

// 일회성 게임 아이템 구매
// 아이템아이디, 개수
export const purchaseGameItem = async (itemId, itemCount) => {
  const url = `${API_BASE_URL}/api/store/items/oneoff`;
  return await axios
    // .post(url, {
    .post(url, { itemId, itemCount }, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      console.log(response.data);
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}

// 대출 아이템 구매
// 대출아이템 아이디
export const purchaseLoanItem = async (loanId) => {
  const url = `${API_BASE_URL}/api/store/items/loan`;
  console.log("token", token);
  return await axios
    // .post(url, {
    .post(url, { loanId }, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}

// 슬롯에 대출 아이템 선택해서 저장
// export const setLoanItems = async (savedSlot) => {
export const setLoanItems = async (slotObject) => {
  const url = `${API_BASE_URL}/api/store/choose-loan`;
  return await axios
    .put(url, slotObject, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      console.log(response.data);
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}

export const getUserInfo = async () => {
  const url = `${API_BASE_URL}/api/user-info`
  return await axios
    .get(url, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      // console.log("user-info:", response.data);
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}