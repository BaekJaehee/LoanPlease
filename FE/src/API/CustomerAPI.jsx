import { Cookies } from "react-cookie";

const cookies = new Cookies();


// API 호출 함수
async function fetchCustomerRequest() {
  const token = cookies.get('Authorization');
  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + 'game/loanrequest', {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch data:", error);
    throw error;  // 에러를 던지거나 에러 상태를 반환할 수 있습니다.
  }
}

async function fetchUserRequest() {
  const token = cookies.get('Authorization');
  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + 'api/user-info', {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch data:", error);
    throw error;  // 에러를 던지거나 에러 상태를 반환할 수 있습니다.
  }
}


async function postScoreRequest(num, data) {
  const token = cookies.get('Authorization');

  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + `game/score/${num}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch data:", error);
    throw error;
  }
}

async function fetchGetawayRequest(data) {
  const token = cookies.get('Authorization');

  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + `game/getaway`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch data:", error);
    throw error;
  }
}

async function patchScoreRequest(score) {
  const token = cookies.get('Authorization');

  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + `game/score?score=${score}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: score
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch data:", error);
    throw error;
  }
}

async function postUseItem(num) {
  const token = cookies.get('Authorization');

  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + `game/use-item`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        userItemId: num
      })
  
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch data:", error);
    throw error;
  }
}


export { fetchCustomerRequest, fetchUserRequest, postScoreRequest, fetchGetawayRequest, patchScoreRequest, postUseItem };