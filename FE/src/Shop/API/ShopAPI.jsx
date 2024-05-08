import axios from "axios";

// 아이템 리스트 조회
// export const itemsList = async (token) => {
export const itemsList = async () => {
  // const url = `/api/store/items`;
  const url = `http://localhost:8080/store/items`;
  return await axios
    .get(url, {
      headers: {
        "Content-Type": "application/json",
        // "Authorization": `Bearer ${token}`,
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
export const purchaseSlot = async (token) => {
  const url = `api/store/items/slot`;
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
export const purchaseGameItem = async (token) => {
  const url = `api/store/items/oneoff`;
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

// 대출 아이템 구매
export const purchaseLoanItem = async (token, loanId) => {
  const url = `api/store/items/loan/${loanId}`;
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

// 슬롯에 대출 아이템 선택
export const setLoanItems = async (token) => {
  const url = `api/store/choose-loan`;
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