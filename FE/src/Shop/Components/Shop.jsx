import { Link } from "react-router-dom";
import { useEffect } from "react";
import useStore from "../../Store/ShopStore";
import { itemsList } from "../API/ShopAPI";
import { getUserInfo } from "../API/ShopAPI";
import GameItem from "./GameItem";
import LoanItem from "./LoanItem";
import SlotSetting from "./SlotSetting";
import SetNumberModal from "../Modal/SetNumberModal";
import GameItemModal from "../Modal/GameItemModal";
import LoanItemModal from "../Modal/LoanItemModal";
import ItemModal from "../Modal/ItemModal";
import WarningModal from "../Modal/WarningModal";
import coin from "../Assets/coin.jpg";
import won from "../Assets/coin_won.png";
import SaveSlotModal from "../Modal/SaveSlotModal";

// 구매여부 0, 1로 구분 -> 구매 확정 시 1로 변환
// 여러번 살 수 있는 아이템?

function Shop() {
  const { 
    setCurrentComponent, currentComponent, 
    setUserInfo, userInfo,
    setPoint, point,
    setSlotNumber, slotNumber,
    setProducts, products,
    setSavedSlot, savedSlot,
    setSelected1, setSelected2, setSelected3, setSelected4, setSelected5, 
    selected1, selected2, selected3, selected4, selected5, 
    isSetNumberModalOpen, openSetNumberModal, closeSetNumberModal,
    isGameItemModalOpen, openGameItemModal, closeGameItemModal, 
    isLoanItemModalOpen, openLoanItemModal, closeLoanItemModal,
    isItemModalOpen, openItemModal, closeItemModal, 
    isSaveSlotModalOpen, openSaveSlotModal, closeSaveSlotModal,
    isWarningModalOpen, openWarningModal, closeWarningModal,
    selectedItem, 
    selectedProduct, 
    gameItems, setGameItems, 
    loanItems, setLoanItems } = useStore();

  const selectedSlots = [selected1, selected2, selected3, selected4, selected5];
  const setSelectedSlots = [setSelected1, setSelected2, setSelected3, setSelected4, setSelected5];
  
  // 아이템 정보 가져오기
  useEffect(() => {
    const fetchItems = async () => {
      try {
        // const token = localStorage.getItem("accessToken");  // 토큰 어떻게 할지는...
        // const data = await itemsList(token); // itemsList 함수를 사용하여 데이터 호출
        const data = await itemsList(); // itemsList 함수를 사용하여 데이터 호출
        // 받은 데이터로 상점 아이템 설정
        setGameItems(data.itemList);
        setLoanItems(data.loanList);
      } catch (error) {
        console.log(error);
      }
    };
    
    fetchItems();
  }, []); // 빈 배열을 전달하여 컴포넌트가 처음 렌더링될 때만 실행되도록 설정

  // 유저 정보 가져오기
  // const fetchUserInfo = async () => {
  //   try {
  //     console.log("시작?")
  //     const data = await getUserInfo();
  //     // console.log("유저정보", data);
  //     console.log("유저정보", data.dataBody);
  //     // 유저 포인트
  //     setPoint(data.dataBody.point);  // 뭐 이런식
  //     // 슬롯 수
  //     setSlotNum(data.dataBody.slotNum);
  //     // 유저가 가진 아이템
  //     setProducts(data.dataBody.userLoanList)
  //   } catch (error) {
  //     console.error(error);
  //   }
  // }

  // useEffect(() => {
  //   fetchUserInfo();
  // }, [point, slotNum, products])


  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const data = await getUserInfo();
        const { point, slotNum, userLoanList, slot_1, slot_2, slot_3, slot_4, slot_5 } = data.dataBody;
        // console.log("유저정보", data);
        console.log("유저정보", data.dataBody);
        // 유저 포인트
        // setPoint(data.dataBody.point);  // 뭐 이런식
        setPoint(point);
        // 슬롯 수
        // setSlotNum(data.dataBody.slotNum);
        setSlotNumber(slotNum);
        // slotNumber = slotNum;
        // 유저가 가진 아이템
        // setProducts(data.dataBody.userLoanList)
        setProducts(userLoanList);
        // 유저가 배치한 슬롯
        // setSavedSlot([data.dataBody.slot_1, data.dataBody.slot_2, data.dataBody.slot_3, data.dataBody.slot_4, data.dataBody.slot_5])
        setSavedSlot([slot_1, slot_2, slot_3, slot_4, slot_5]);
        // setSelected1(products.loanName[0])
      } catch (error) {
        console.error(error);
      }
    };
    fetchUserInfo();
    // 유저 정보 상태에 변화가 있을 때만 실행
  // }, [point, slotNum, products]);
  }, []);

  useEffect(() => {
    // savedSlot에 있는 숫자를 이용하여 해당하는 loanName을 찾아 selectedSlots에 설정
    const updatedSelectedSlots = savedSlot.map(slotId => {
      // savedSlot에 있는 각 숫자가 loanId와 일치하는 상품을 찾아 selectedSlots에 설정
      const selectedProduct = products.find(product => product.loanId === slotId);
      // 해당하는 상품이 없다면 빈 문자열로 설정
      return selectedProduct ? selectedProduct.loanName : "";
    });
  
    // SlotSetting 컴포넌트로 selectedSlots를 전달
    setSelected1(updatedSelectedSlots[0]);
    setSelected2(updatedSelectedSlots[1]);
    setSelected3(updatedSelectedSlots[2]);
    setSelected4(updatedSelectedSlots[3]);
    setSelected5(updatedSelectedSlots[4]);
  }, [savedSlot, products]);

  // useEffect(() => {
  //   const fetchUserInfo = async () => {
  //     try {
  //       const data = await getUserInfo();
  //       const { point, slotNum, userLoanList, slot_1, slot_2, slot_3, slot_4, slot_5 } = data.dataBody;
  //       console.log("유저정보", data.dataBody);
  //       // 새로운 변수에 유저 정보 할당
  //       const newPoint = point;
  //       const newSlotNum = slotNum;
  //       const newUserLoanList = userLoanList;
  //       const newSavedSlot = [slot_1, slot_2, slot_3, slot_4, slot_5];
  
  //       // 상태 업데이트
  //       setPoint(newPoint);
  //       setSlotNumber(newSlotNum);
  //       setProducts(newUserLoanList);
  //       setSavedSlot(newSavedSlot);
  //     } catch (error) {
  //       console.error(error);
  //     }
  //   };
  //   fetchUserInfo();
  // }, []); // 의존성 배열이 비어있으므로 컴포넌트가 처음 마운트될 때만 실행


  // console.log("Shop", slotNum);

  // const handleUpdatePoint = (updatedPoint) => {
  //   setUserPoint(updatedPoint);
  // }
  
  let currentPage;
  // 유저 포인트 props로 전달하기 -> 포인트 부족 시 warning modal
  if (currentComponent === "gameItem") {
    // currentPage = <GameItem openSetNumberModal={openSetNumberModal} openGameItemModal={openGameItemModal} openWarningModal={openWarningModal} gameItems={gameItems} userPoint={userPoint} setUserPoint={handleUpdatePoint} />;
    currentPage = <GameItem openSetNumberModal={openSetNumberModal} openGameItemModal={openGameItemModal} openWarningModal={openWarningModal} gameItems={gameItems} point={point} slotNumber={slotNumber} />;
  } else if (currentComponent == "loanItem") {
    // currentPage = <LoanItem openLoanItemModal={openLoanItemModal} openWarningModal={openWarningModal} loanItems={loanItems} userPoint={userPoint} setUserPoint={handleUpdatePoint} />;
    currentPage = <LoanItem openLoanItemModal={openLoanItemModal} openWarningModal={openWarningModal} loanItems={loanItems} point={point} />;
  } else {
    currentPage = <SlotSetting openItemModal={openItemModal} openSaveSlotModal={openSaveSlotModal} products={products} slotNumber={slotNumber} savedSlot={savedSlot} selectedSlots={selectedSlots} />;
  }

  return (
    <>
      {isSetNumberModalOpen && selectedItem && <SetNumberModal closeSetNumberModal={closeSetNumberModal} openGameItemModal={openGameItemModal} openWarningModal={openWarningModal} point={point} itemId={selectedItem.itemId} price={selectedItem.price} />}
      {/* {isSetNumberModalOpen && <SetNumberModal closeSetNumberModal={closeSetNumberModal} openGameItemModal={openGameItemModal} openWarningModal={openWarningModal} userPoint={userPoint} itemId={selectedItem.itemId} price={selectedItem.price} />} */}
      {isGameItemModalOpen && <GameItemModal closeGameItemModal={closeGameItemModal} />}
      {isLoanItemModalOpen && <LoanItemModal closeLoanItemModal={closeLoanItemModal} />}
      {isItemModalOpen && <ItemModal closeItemModal={closeItemModal} selectedProduct={selectedProduct} />}
      {isWarningModalOpen && <WarningModal closeWarningModal={closeWarningModal} closeSetNumberModal={closeSetNumberModal} />}
      {isSaveSlotModalOpen && <SaveSlotModal closeSaveSlotModal={closeSaveSlotModal} />}
      <div className="bg-cusColor3 min-h-screen w-full flex">
        <img src={coin} alt="배경" className="absolute w-full h-full object-cover opacity-50 z-0" />
        {/* 사이드바 */}
        {/* <div className="w-1/5 border-r-2 flex flex-col font-cusFont1"> */}
        <div className="w-[250px] border-r-2 font-cusFont1 relative z-10">
          <div className="h-[15%]"></div>
          {/* <div className="p-4 flex-1 mt-20"> */}
          <div className="p-4 flex-1">
            {/* 게임진행 아이템 */}
            <button
              // className="block w-full mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg mb-2 border-2"
              // className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg mb-2 border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg mb-2 border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("gameItem")}
            >
              게임진행
            </button>
            {/* 대출상품 아이템 */}
            <button
              // className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("loanItem")}
            >
              대출상품
            </button>
            {/* 대출상품 슬롯 세팅 */}
            <button
              // className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
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
          {/* <div className="border-b-0 p-4 flex justify-end items-center"> */}
          <div className="border-b-0 p-4 flex justify-end items-center h-[15%]">
            <div className="border-2 rounded-lg bg-white text-right font-cusFont1 text-xl mx-6 pl-2 pr-4 py-2 w-[180px] border-black flex items-center justify-between">
              <img src={won} alt="아이콘" className="w-7 h-7" />
              <span>{point}</span>
            </div>
          </div>
          {/* 아이템 */}
          {/* <div className="p-4"> */}
          <div className="p-4 h-[85%]">
            {currentPage}
          </div>
        </div>
      </div>
    </>
  );
}

export default Shop;
