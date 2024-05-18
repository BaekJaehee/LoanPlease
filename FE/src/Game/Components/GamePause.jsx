import useStore from "../../Store/GameStore.jsx"
import { useNavigate } from 'react-router-dom';

function GamePause () {

  const navigate = useNavigate();
  const { score, isGamePause, setGamePause, resetGame } = useStore();

  const handleRestart = () => {
    window.location.reload();
  };

  const handleGoHome = () => {
    resetGame()
    window.location.href = '/';
  };

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-gray-800 bg-opacity-75 text-center">
      <div className="p-5 h-[350px] w-[600px] bg-cusColor1 rounded shadow-lg text-white border-[15px] border-cusColor3">
        <div className="h-[40px]"></div>
        <p className="text-[40px]">정말 그만두시겠습니까?</p>
        <br />
        <div className="flex justify-between items-center mt-[50px]">
          <button  onClick={setGamePause} className="ml-[5%] w-[40%] h-[60px] bg-cusColor5 rounded-lg">
            <p className="text-4xl text-cusColor4">계속하기</p>
          </button>
          <button  onClick={handleGoHome} className="mr-[5%] w-[40%] h-[60px] bg-red-500 rounded-lg">
            <p className="text-4xl">홈으로</p>
          </button>
        </div>
      </div>
    </div>
  );
}

export default GamePause