import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useStore from "../../Store/GameStore.jsx"

import Exit from "../Assets/exit.png"
import Help from "../Assets/help.png"


function GameStart() {

  const navigate = useNavigate()

  const handleGoHome = () => {
    navigate('/');
  };

  const { startTimer, showModal, setShowModal, loadUserInfo } = useStore();
  const [count, setCount] = useState(null);

  useEffect(() => {
    let timerId = null;
    if (count > 0) {
      // 카운트다운을 위한 타이머 설정
      timerId = setTimeout(() => {
        setCount(count - 1);
      }, 1000);
    } else if (count === 0) {
      // 카운트다운이 0에 도달하면 타이머 시작
      startTimer();
      loadUserInfo();
    }
    return () => clearTimeout(timerId); // 컴포넌트 언마운트 시 타이머 정리
  }, [count, startTimer]);

  const handleStartClick = () => {
    if (count === null) {
      setCount(3); // 버튼 클릭 시 카운트다운 시작
    }
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-gray-800 bg-opacity-75">
      {count !== null ? (
        <div className="p-5 bg-transparent rounded">
          <p className="text-[80px] text-white">{count}</p>
        </div>
      ) : (
        <>
          <button className='absolute top-[14px] right-[100px] w-[78px] h-[60px]' onClick={setShowModal}>
            <img src={Help} alt="" />
          </button>
          <button className='absolute top-7 right-3 w-[60px] h-[60px]' onClick={handleGoHome}>
            <img src={Exit} alt="" />
          </button>
          <button onClick={handleStartClick} className="mt-4 bg-blue-500 text-white rounded-lg p-4 hover:bg-blue-700 text-4xl">
            게임 시작하기
          </button>
        </>
      )}
    </div>
  );
}

export default GameStart