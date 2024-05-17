import useStore from "../../Store/GameStore.jsx"

function GameInfo() {
  const { gameInfo } = useStore();

  const { customerInfo, financialInfo } = gameInfo;

  return (
    <div className="font-cusFont2 ml-5 mt-2">
      <p className="text-[25px] font-cusFont1 mb-1">기본정보</p>
      {customerInfo ? (
        <div className="mb-2 ml-5 text-[20px]">
          <p>이름: {customerInfo.name}</p>
          <p>나이: {customerInfo.age}</p>
          <p>성별: {customerInfo.gender}</p>
          {/* <p>메세지: {customerInfo.customerMessage}</p> */}
        </div>
      ) : (
        <p>No customer information available.</p>
      )}

      <p className="text-[25px] font-cusFont1 mb-1">금융정보</p>
      {financialInfo ? (
        <div className="ml-5 text-[20px]">
          <p>연봉: {financialInfo.incomeTotal}만원</p>
          <p>직종: {financialInfo.incomeType}</p>
          <p>직업: {financialInfo.occypType}</p>
          {/* 다른 필요한 속성을 추가로 렌더링 */}
        </div>
      ) : (
        <p>No financial information available.</p>
      )}
    </div>
  );
}

export default GameInfo;
