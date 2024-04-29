// 현재 로그인한 유저의 슬롯 개수랑 갖고있는 상품을 보여줌
// 체크박스? 드래그? 아니면 1번 슬롯 클릭한 상태에서 상품 클릭하면 그게 1번으로 들어가도록?
// -> 3으로 해봄 클릭하면 테두리 색깔 다르게해서
// 저장버튼 필요?
// 보유슬롯 몇개인지, 상품 몇개고 뭐가 슬롯에 들어있는지 표시 필요(DB에서)

function SlotSetting() {
  return(
    <div>
      <div className="border-2 bg-white">
        구매한 대출상품들 슬롯에 세팅
      </div>
      <div className={`flex-grow-1 w-[280px] border-2 px-6 py-4 rounded-lg border-black`}>
          <p className="font-cusFont1 text-3xl py-4 my-2">사용자가 가지고있는 아이템 이름</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">설명</p>
          </div>
        </div>
    </div>
  );
}

export default SlotSetting;