import useStore from "../../Store/GameStore.jsx"
import { useNavigate } from 'react-router-dom';

function GameEnd () {

  const navigate = useNavigate();
  const { score } = useStore();

  const handleRestart = () => {
    window.location.reload();
  };

  const handleGoHome = () => {
    navigate('/');
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-gray-800 bg-opacity-75">
      <div className="p-5 h-[350px] w-[600px] bg-cusColor1 rounded shadow-lg text-white border-[15px] border-cusColor3">
        <p className="text-[40px]">퇴근하세요!</p>
        <br />
        <p className="mt-[30px] text-4xl">SCORE : {score}</p>
        <div className="flex justify-between items-center mt-[50px]">
          <button  onClick={handleRestart} className="ml-[5%] w-[40%] h-[60px] bg-cusColor5 rounded-lg">
            <p className="text-4xl text-cusColor4">다시하기</p>
          </button>
          <button  onClick={handleGoHome} className="mr-[5%] w-[40%] h-[60px] bg-red-500 rounded-lg">
            <p className="text-4xl">홈으로</p>
          </button>
        </div>
      </div>
    </div>
  );
}

export default GameEnd