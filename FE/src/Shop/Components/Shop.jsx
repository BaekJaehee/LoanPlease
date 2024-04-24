import { Link } from "react-router-dom";
import useStore from "../../Store/ShopStore";
import GameItem from "./GameItem";
import LoanItem from "./LoanItem";
import ShopModal from "../Modal/ShopModal";
import coin from "../Assets/coin.jpg";

function Shop() {
  const { setCurrentComponent, currentComponent, isModalOpen, openModal, closeModal } = useStore();

  let currentPage;
  if (currentComponent === "gameItem") {
    currentPage = <GameItem openModal={openModal} />;
  } else {
    currentPage = <LoanItem openModal={openModal} />;
  }

  return (
    <>
      {isModalOpen && <ShopModal closeModal={closeModal} />}
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
          </div>
          <div className="p-4 absolute bottom-2">
            {/* 클릭하면 메인 페이지로 */}
            <Link to="/">
              <button className="block w-[215px] py-2 text-white border-2 border-b-4 bg-blue-800 hover:bg-blue-900 rounded-lg self-end text-lg border-black">
                돌아가기
              </button>
            </Link>
          </div>
        </div>
        {/* 콘텐츠 */}
        <div className="flex-1 z-10">
          {/* 잔액 */}
          <div className="border-b-0 p-4 flex justify-end items-center">
            <div className="border-2 rounded-lg bg-white text-right font-cusFont1 text-lg mx-6 pl-4 pr-4 py-2 w-[180px] border-black">
              1000
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
