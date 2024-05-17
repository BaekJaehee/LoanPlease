import { Cookies } from "react-cookie";

const cookies = new Cookies();


// API 호출 함수
async function fetchCustomerRequest() {
  const token = cookies.get('Authorization');
  try {
    const response = await fetch('https://loanplease.kr/game/loanrequest', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    
    });
    if (!response.ok) {
      console.log(response)
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
    const response = await fetch('https://loanplease.kr/api/user-info', {
      headers: {
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
    const response = await fetch(`https://loanplease.kr/game/score/${num}`, {
      method: 'POST',
      headers: {
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
    const response = await fetch(`https://loanplease.kr/game/getaway`, {
      method: 'POST',
      headers: {
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
    const response = await fetch(`https://loanplease.kr/game/score?score=${score}`, {
      method: 'PATCH',
      headers: {
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
    const response = await fetch(`https://loanplease.kr/game/use-item`, {
      method: 'POST',
      headers: {
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