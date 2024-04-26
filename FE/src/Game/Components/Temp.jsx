import axios from "axios";
import React, { useState, useEffect } from 'react';

function Temp() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get("https://loanplease.kr/api/test");
        setData(response.data);
      } catch (error) {
        console.error('API 요청 중 오류 발생:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      <h1>테스트</h1>
      <div>Data from API: {data ? JSON.stringify(data) : 'Loading...'}</div>
    </div>
  );
}

export default Temp;
