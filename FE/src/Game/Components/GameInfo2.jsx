import useStore from "../../Store/GameStore.jsx"

function GameInfo2() {
  const { gameInfo } = useStore();

  const { customerInfo, financialInfo, nonFinancialInfo } = gameInfo;

  return (
    <div className="font-cusFont2 mt-3 ml-5">
      <p className="text-[25px] font-cusFont1 mb-1">비금융정보</p>
      {financialInfo ? (
        <div className="ml-5 text-[20px]">
          <p>자차여부: {nonFinancialInfo.car}</p>
          <p>최종 학력: {nonFinancialInfo.eduType}</p>
          {/* <p>근로일수: {nonFinancialInfo.employedDays}</p> */}
          <p>주거 형태: {nonFinancialInfo.houseType}</p>
          <p>부동산 자가 여부: {nonFinancialInfo.reality}</p>
          <p>가족 수: {nonFinancialInfo.familySize}</p>
          <p>가족 유형: {nonFinancialInfo.familyType}</p>
          <p>자녀 수: {nonFinancialInfo.childNum}</p>
        </div>
      ) : (
        <p>No financial information available.</p>
      )}
    </div>
  );
}

export default GameInfo2;
