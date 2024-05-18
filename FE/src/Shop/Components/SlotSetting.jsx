import { PropTypes } from "prop-types"; 
import { useEffect } from "react";
import useStore from "../../Store/ShopStore";
import { setLoanItems } from "../API/ShopAPI";

function SlotSetting({ openItemModal, openSaveSlotModal, products, slotNumber, savedSlot, selectedSlots, setSelectedSlots, openWarningModal2 }) {
  const { 
    currentPage, setCurrentPage, 
    setSelectedProduct, 
    setSavedSlot, 
    setSelected1, setSelected2, setSelected3, setSelected4, setSelected5 
  } = useStore();
  // const [originalSelectedSlots, setOriginalSelectedSlots] = useState([]);

  // const selectedSlots = [selected1, selected2, selected3, selected4, selected5];
  // const setSelectedSlots = [setSelected1, setSelected2, setSelected3, setSelected4, setSelected5];

  const itemsPerPage = 5; // 페이지당 보여줄 아이템 수(임의)
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = products.slice(indexOfFirstItem, indexOfLastItem);
  
  useEffect(() => {
    setCurrentPage(1); // 페이지가 변경될 때마다 첫 페이지로 초기화
  }, [products, setCurrentPage]);

  // useEffect(() => {
  //   // 초기 selectedSlots을 originalSelectedSlots에 저장
  //   setOriginalSelectedSlots(selectedSlots.map(slot => ({ ...slot }))); // selectedSlots 배열 복사
  // }, [selectedSlots]);

  const showDetail = (content) => {
    setSelectedProduct(content);
  };

  const setting = (item) => {
    // 이미 배치된건지 검사
    const isAlreadySelected = selectedSlots.some(selectedSlot => selectedSlot.name === item.name);
    console.log(isAlreadySelected);
    // 중복이 아닐 때
    if (!isAlreadySelected) {
      // 선택된 슬롯을 찾아 상태를 업데이트
      for (let i = 0; i < slotNumber; i++) {
        if (!selectedSlots[i].name) {
          setSelectedSlots[i](item);
          savedSlot[i] = item.id;  // [0, 0, 0, 0, 0]에서 바꾸기
          // setSelectedSlots(prevSlots => {
          //   const updatedSlots = [...prevSlots];
          //   updatedSlots[i] = item; // item을 직접 넣어야 함
          //   return updatedSlots;
          // });
          // setSavedSlot(prevSavedSlot => {
          //   const updatedSavedSlot = [...prevSavedSlot];
          //   updatedSavedSlot[i] = item.id; // item의 id를 넣어야 함
          //   return updatedSavedSlot;
          // });
          break;
        }
      }
    }
  };
  
  const clear = (selected, clearFunction) => {
    // 선택된 슬롯의 인덱스를 찾기
    const selectedIndex = selectedSlots.findIndex(slot => slot === selected);
    // 현재 클릭한 슬롯을 비움(화면)
    // clearFunction({ name: null, description: null });
    clearFunction({ name: null });
    // 슬롯 데이터 비움(0)
    savedSlot[selectedIndex] = 0;
  
    // 앞쪽 슬롯들을 앞으로 당겨오기
    for (let i = selectedIndex; i < slotNumber - 1; i++) {
      setSelectedSlots[i](selectedSlots[i + 1]);
      // 슬롯을 비우고 0으로 채움
      savedSlot[i] = savedSlot[i + 1] ? savedSlot[i + 1] : 0;
    }
    // 마지막 슬롯을 비움
    setSelectedSlots[slotNumber - 1]({ name: null });
    // 마지막 슬롯을 0으로 채움
    savedSlot[slotNumber - 1] = 0;
  
    // savedSlot 상태 업데이트
    setSavedSlot([...savedSlot]);
  };

  // 슬롯 리셋
  const reset = () => {
    // 화면에서 지우기
    setSelected1({ name: null });
    setSelected2({ name: null });
    setSelected3({ name: null });
    setSelected4({ name: null });
    setSelected5({ name: null });
    // 배열 초기화
    setSavedSlot([0, 0, 0, 0, 0]);
  }

  // 슬롯 저장
  // const save = async () => {
  //   try {
  //     console.log(savedSlot);
  //     const slotObject = {
  //       "slot_1": savedSlot[0],
  //       "slot_2": savedSlot[1],
  //       "slot_3": savedSlot[2],
  //       "slot_4": savedSlot[3],
  //       "slot_5": savedSlot[4],
  //     };
  //     console.log("저장", slotObject)
  //     // await setLoanItems(savedSlot)
  //     await setLoanItems(slotObject)
  //     console.log(slotObject)
  //   } catch (error) {
  //     console.error(error);
  //   }
  // }

  // 슬롯 저장
  const handleSaveSlot = async (savedSlot) => {
    if (savedSlot.filter(slot => slot !== 0).length < 3) {
      openWarningModal2();
      // 실패 시 원래의 selectedSlots 상태로 복원
      // setSelectedSlots(originalSelectedSlots.map(slot => ({ ...slot })));
      return;
    }

    try {
      const slotObject = {
        "slot_1": savedSlot[0],
        "slot_2": savedSlot[1],
        "slot_3": savedSlot[2],
        "slot_4": savedSlot[3],
        "slot_5": savedSlot[4],
      };
      openSaveSlotModal();
      // 저장 성공 시 originalSelectedSlots 업데이트
      // setOriginalSelectedSlots(selectedSlots.map(slot => ({ ...slot })));
      await setLoanItems(slotObject)
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <div className="h-full">
      {/* 페이지네이션 */}
      <div className="flex justify-center my-2 font-cusFont1 h-[8%]">
        <button 
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform h-[45px]"
          onClick={() => setCurrentPage(currentPage - 1)} 
          disabled={currentPage === 1} 
        >
          이전 페이지
        </button>
        <button 
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform h-[45px]"
          onClick={() => setCurrentPage(currentPage + 1)} 
          disabled={currentItems.length !== itemsPerPage || currentPage === Math.ceil(products.length / itemsPerPage)} 

        >
          다음 페이지
        </button>
      </div>
      {/* 사용자가 가지고 있는 아이템(products) */}
      <div className="flex justify-stretch gap-3 h-[50%]">
        {/* {products.map((item, index) => ( */}
        {currentItems.map((item, index) => (
          <div 
          key={index} 
          className="font-cusFont1 flex-grow-1 w-[240px] h-[95%] border-2 px-6 py-4 rounded-lg border-black bg-white text-center"
          style={{ backgroundColor: item.color }}
          >
            <p className="text-2xl mx-3 py-4 my-2 h-[30%] place-content-center">{item.loanName}</p>
            <div className="h-[55%] text-xl py-3 my-2 place-content-center">
              <button 
                className="mx-2 px-4 py-2 bg-amber-300 hover:bg-amber-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
                onClick={() => {
                  showDetail(item);
                  openItemModal();
                }}
              >
                상세보기
              </button>
              <button 
                className="mx-2 px-4 py-2 bg-blue-300 hover:bg-blue-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
                onClick={() => setting({ id: item.loanId, name: item.loanName })}
              >
                선택하기
              </button>
            </div>
          </div>
        ))}
      </div>
      {/* 소지 슬롯개수(slotNumber)만큼 */}
      <div className="flex justify-center gap-4 h-[20%]">
      {[...Array(slotNumber)].map((_, index) => {
          const selectedSlot = selectedSlots[index];
          const setSelectedSlot = setSelectedSlots[index];
          return (
            <div
              key={index}
              className="flex-grow-1 w-[220px] h-[90%] border-2 px-6 py-4 rounded-lg border-black bg-white hover:cursor-pointer text-center place-content-center"
              onClick={() => clear(selectedSlot, setSelectedSlot)}
            >
              {/* 여기에 클릭해서 넣은거 표시되어야함 */}
              <p className="font-cusFont1 text-2xl py-2 my-2">{selectedSlot.name}</p>
            </div>
          );
        })}
      </div>
      <div className="flex justify-center h-[10%]">
        {/* 슬롯 저장 */}
        <button 
          onClick={() => {
            handleSaveSlot(savedSlot);
          }} 
          className="w-[100px] h-[50px] transform -translate-x-1/2 font-cusFont1 mx-2 px-4 py-2 bg-blue-300 hover:bg-blue-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
        >
          저장
        </button>
        {/* 슬롯 초기화 */}
        <button 
          onClick={reset} 
          className="w-[100px] h-[50px] transform -translate-x-1/2 font-cusFont1 mx-2 px-4 py-2 bg-red-300 hover:bg-red-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
        >
          초기화
        </button>
      </div>
    </div>
  );
}

SlotSetting.propTypes = {
  openItemModal: PropTypes.func.isRequired,
  openSaveSlotModal: PropTypes.func.isRequired,
  products: PropTypes.array.isRequired,
  slotNumber: PropTypes.number.isRequired,
  savedSlot: PropTypes.array.isRequired,
  selectedSlots: PropTypes.array.isRequired,
  setSelectedSlots: PropTypes.array.isRequired,
  openWarningModal2: PropTypes.func.isRequired,
}

export default SlotSetting;
