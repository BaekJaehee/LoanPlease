import axios from "axios";

// 192.168.100.178:8080

const token = document.cookie // 쿠키 읽기
      .split('; ')  // 항목 분리
      .find(row => row.startsWith('Authorization='))  // Authorization 찾기
      ?.split('=')[1];  // 의 value 값

// 아이템 리스트 조회
// export const itemsList = async (token) => {
export const itemsList = async () => {
  // const url = `/api/store/items`;
  const url = `http://localhost:8080/store/items`;
  // const url = `http://192.168.100.178:8080/store/items`;
  return await axios
    .get(url, {
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
  };

// 슬롯 구매
// 유저 슬롯개수 +1, 포인트 차감
// export const purchaseSlot = async (token) => {
export const purchaseSlot = async () => {
  // const url = `api/store/items/slot`;
  const url = `http://localhost:8080/store/items/slot`;
  return await axios
    .post(url, {
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

// 일회성 게임 아이템 구매
// 아이템아이디, 개수
// export const purchaseGameItem = async (token) => {
export const purchaseGameItem = async (itemId, number) => {
  // const url = `api/store/items/oneoff`;
  const url = `http://localhost:8080/store/items/oneoff`;
  return await axios
    // .post(url, {
    .post(url, { itemId, number }, {
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

// 대출 아이템 구매
// 대출아이템 아이디
// export const purchaseLoanItem = async (token, loanId) => {
export const purchaseLoanItem = async (loanId) => {
  // const url = `api/store/items/loan/${loanId}`;
  // const url = `http://localhost:8080/store/items/loan/${loanId}`;
  const url = `http://localhost:8080/store/items/loan`;
  return await axios
    // .post(url, {
    .post(url, loanId, {
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

// 슬롯 세팅 페이지 들어가면 유저가 보유한 대출 아이템 가져옴
export const owendLoanItems = async () => {
  const url = `http://localhost:8080/store/setting`;
  return await axios
    .get(url, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      // console.log("test", response.data)
      // const data = response.data.map(item => item.loan);
      // console.log(data);
      // return data;
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    })
}

// 슬롯에 대출 아이템 선택해서 저장
// export const setLoanItems = async (token) => {
export const setLoanItems = async (savedSlot) => {
// export const setLoanItems = async (jsonData) => {
  // const url = `api/store/choose-loan`;
  const url = `http://localhost:8080/store/choose-loan`;
  return await axios
    // .post(url, jsonData, {
    // .post(url, { slotIds: savedSlot }, {
    .post(url, savedSlot, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      console.log("setLoanItems:", response.data)
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}

export const getUserInfo = async () => {
  const url = `http://192.168.100.178:8080/api/user-info`
  return await axios
    .get(url, {
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    })
    .then((response) => {
      console.log("user-info:", response.data)
      return response.data;
    })
    .catch((e) => {
      console.log(e);
    });
}