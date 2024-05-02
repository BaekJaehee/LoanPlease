import { Link } from "react-router-dom";
import useStore from "../../Store/ShopStore";
import GameItem from "./GameItem";
import LoanItem from "./LoanItem";
import SlotSetting from "./SlotSetting";
import ShopModal from "../Modal/ShopModal";
import ItemModal from "../Modal/ItemModal";
import coin from "../Assets/coin.jpg";
import won from "../Assets/coin_won.png";

// 구매여부 0, 1로 구분 -> 구매 확정 시 1로 변환
// 여러번 살 수 있는 아이템?

function Shop() {
  const { setCurrentComponent, currentComponent, isShopModalOpen, openShopModal, closeShopModal, isItemModalOpen, openItemModal, closeItemModal, gameItems, loanItems, product } = useStore();

  let currentPage;
  if (currentComponent === "gameItem") {
    currentPage = <GameItem openShopModal={openShopModal} />;
    // currentPage = gameItems.map((item, index) => (
    //   <GameItem key={index} item={item} openModal={openModal} />
    // ));
    
  } else if (currentComponent == "loanItem") {
    currentPage = <LoanItem openShopModal={openShopModal} />;
    // currentPage = loanItems.map((item, index) => (
    //   <LoanItem key={index} item={item} openModal={openModal} />
    // ));
  } else {
    currentPage = <SlotSetting openItemModal={openItemModal} />;
  }

  return (
    <>
      {isShopModalOpen && <ShopModal closeShopModal={closeShopModal} />}
      {isItemModalOpen && <ItemModal closeItemModal={closeItemModal} />}
      <div className="bg-cusColor3 min-h-screen w-full flex">
        <img src={coin} alt="배경" className="absolute w-full h-full object-cover opacity-50 z-0" />
        {/* 사이드바 */}
        {/* <div className="w-1/5 border-r-2 flex flex-col font-cusFont1"> */}
        <div className="w-[250px] border-r-2 font-cusFont1 relative z-10">
          <div className="p-4 flex-1 mt-20">
            {/* 게임진행 아이템 */}
            <button
              // className="block w-full mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg mb-2 border-2"
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg mb-2 border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("gameItem")}
            >
              게임진행
            </button>
            {/* 대출상품 아이템 */}
            <button
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("loanItem")}
            >
              대출상품
            </button>
            {/* 대출상품 슬롯 세팅 */}
            <button
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("slotSetting")}
            >
              슬롯세팅
            </button>
          </div>
          <div className="p-4 absolute bottom-2">
            {/* 클릭하면 메인 페이지로 */}
            <Link to="/">
              <button className="block w-[215px] py-2 text-white border-2 border-b-4 bg-blue-800 hover:bg-blue-900 rounded-lg self-end text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform">
                돌아가기
              </button>
            </Link>
          </div>
        </div>
        {/* 콘텐츠 */}
        <div className="flex-1 z-10">
          {/* 잔액 */}
          <div className="border-b-0 p-4 flex justify-end items-center">
            <div className="border-2 rounded-lg bg-white text-right font-cusFont1 text-xl mx-6 pl-2 pr-4 py-2 w-[180px] border-black flex items-center justify-between">
              <img src={won} alt="아이콘" className="w-7 h-7" />
              <span>1000</span>
            </div>
          </div>
          {/* 아이템 */}
          <div className="p-4">
            {currentPage}
          </div>
        </div>
      </div>
    </>
  );
}

export default Shop;
