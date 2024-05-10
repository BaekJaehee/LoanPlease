// API 호출 함수
async function fetchCustomerRequest() {
  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + 'game/loanrequest');
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
  
  try {
    const response = await fetch(import.meta.env.VITE_BASE_URL + `game/score/${num}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
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


export { fetchCustomerRequest, postScoreRequest };