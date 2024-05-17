import { Link } from "react-router-dom";
import { useEffect } from "react";
import useStore from "../../Store/ShopStore";
import { itemsList, getUserInfo } from "../API/ShopAPI";
import GameItem from "./GameItem";
import LoanItem from "./LoanItem";
import SlotSetting from "./SlotSetting";
import SetNumberModal from "../Modal/SetNumberModal";
import GameItemModal from "../Modal/GameItemModal";
import LoanItemModal from "../Modal/LoanItemModal";
import ItemModal from "../Modal/ItemModal";
import WarningModal from "../Modal/WarningModal";
import WarningModal2 from "../Modal/WarningModal2";
import coin from "../Assets/coin.jpg";
import won from "../Assets/coin_won.png";
import SaveSlotModal from "../Modal/SaveSlotModal";

function Shop() {
  const {
    setCurrentComponent, currentComponent,
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
    isWarningModal2Open, openWarningModal2, closeWarningModal2,
    selectedItem,
    selectedProduct,
    gameItems, setGameItems,
    loanItems, setLoanItems
  } = useStore();

  const selectedSlots = [selected1, selected2, selected3, selected4, selected5];
  const setSelectedSlots = [setSelected1, setSelected2, setSelected3, setSelected4, setSelected5];

  useEffect(() => {
    const fetchItems = async () => {
      try {
        const data = await itemsList();
        setGameItems(data.itemList);
        setLoanItems(data.loanList);
      } catch (error) {
        console.log(error);
      }
    };
    fetchItems();
  }, []);

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const data = await getUserInfo();
        const { point, slotNum, userLoanList, slot_1, slot_2, slot_3, slot_4, slot_5 } = data.dataBody;
        setPoint(point);
        setSlotNumber(slotNum);
        setProducts(userLoanList);
        setSavedSlot([slot_1, slot_2, slot_3, slot_4, slot_5]);
        
        const initialSelectedSlots = [
          slot_1, slot_2, slot_3, slot_4, slot_5
        ].map(slotId => {
          const loan = userLoanList.find(product => product.loanId === slotId);
          return loan ? loan.loanName : null;
        });
        const initialSelectedSlotsColor = [
          slot_1, slot_2, slot_3, slot_4, slot_5
        ].map(slotId => {
          const loan = userLoanList.find(product => product.loanId === slotId);
          return loan ? loan.color : null;
        });
        setSelected1({ name: initialSelectedSlots[0], color: initialSelectedSlotsColor[0] });
        setSelected2({ name: initialSelectedSlots[1], color: initialSelectedSlotsColor[1] });
        setSelected3({ name: initialSelectedSlots[2], color: initialSelectedSlotsColor[2] });
        setSelected4({ name: initialSelectedSlots[3], color: initialSelectedSlotsColor[3] });
        setSelected5({ name: initialSelectedSlots[4], color: initialSelectedSlotsColor[4] });
      } catch (error) {
        console.error(error);
      }
    };
    fetchUserInfo();
  }, []);

  let currentPage;
  if (currentComponent === "gameItem") {
    currentPage = <GameItem openSetNumberModal={openSetNumberModal} openGameItemModal={openGameItemModal} openWarningModal={openWarningModal} gameItems={gameItems} point={point} slotNumber={slotNumber} />;
  } else if (currentComponent === "loanItem") {
    currentPage = <LoanItem openLoanItemModal={openLoanItemModal} openWarningModal={openWarningModal} loanItems={loanItems} point={point} products={products} />;
  } else {
    currentPage = <SlotSetting openItemModal={openItemModal} openSaveSlotModal={openSaveSlotModal} products={products} slotNumber={slotNumber} savedSlot={savedSlot} selectedSlots={selectedSlots} setSelectedSlots={setSelectedSlots} openWarningModal2={openWarningModal2} />;
  }

  return (
    <>
      {isSetNumberModalOpen && selectedItem && <SetNumberModal closeSetNumberModal={closeSetNumberModal} openGameItemModal={openGameItemModal} openWarningModal={openWarningModal} point={point} itemId={selectedItem.itemId} price={selectedItem.price} />}
      {isGameItemModalOpen && <GameItemModal closeGameItemModal={closeGameItemModal} />}
      {isLoanItemModalOpen && <LoanItemModal closeLoanItemModal={closeLoanItemModal} />}
      {isItemModalOpen && <ItemModal closeItemModal={closeItemModal} selectedProduct={selectedProduct} />}
      {isWarningModalOpen && <WarningModal closeWarningModal={closeWarningModal} closeSetNumberModal={closeSetNumberModal} />}
      {isSaveSlotModalOpen && <SaveSlotModal closeSaveSlotModal={closeSaveSlotModal} />}
      {isWarningModal2Open && <WarningModal2 closeWarningModal2={closeWarningModal2} />}
      <div className="bg-cusColor3 min-h-screen w-full flex">
        <img src={coin} alt="배경" className="absolute w-full h-full object-cover opacity-50 z-0" />
        <div className="w-[250px] border-r-2 font-cusFont1 relative z-10">
          <div className="h-[15%]"></div>
          <div className="p-4 flex-1">
            <button
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg mb-2 border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("gameItem")}
            >
              게임진행
            </button>
            <button
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("loanItem")}
            >
              대출상품
            </button>
            <button
              className="block w-[215px] mb-4 py-2 bg-orange-400 hover:bg-orange-600 rounded-lg border-2 border-b-4 text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
              onClick={() => setCurrentComponent("slotSetting")}
            >
              슬롯세팅
            </button>
          </div>
          <div className="p-4 absolute bottom-2">
            <Link to="/">
              <button className="block w-[215px] py-2 text-white border-2 border-b-4 bg-blue-800 hover:bg-blue-900 rounded-lg self-end text-lg border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform">
                돌아가기
              </button>
            </Link>
          </div>
        </div>
        <div className="flex-1 z-10">
          <div className="border-b-0 p-4 flex justify-end items-center h-[15%]">
            <div className="border-2 rounded-lg bg-white text-right font-cusFont1 text-xl mx-6 pl-2 pr-4 py-2 w-[180px] border-black flex items-center justify-between">
              <img src={won} alt="아이콘" className="w-7 h-7" />
              <span>{point}</span>
            </div>
          </div>
          <div className="p-4 h-[85%]">
            {currentPage}
          </div>
        </div>
      </div>
    </>
  );
}

export default Shop;
