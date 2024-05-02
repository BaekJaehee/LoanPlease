import { PropTypes } from "prop-types"; 
import { useEffect } from "react";
import useStore from "../../Store/ShopStore";

// 슬롯에 중복으로 못넣게 하기

function SlotSetting({ openItemModal }) {
  const { products, userSlotNum, currentPage, setCurrentPage, setSelectedProduct, savedSlot, setSavedSlot, selected1, selected2, selected3, selected4, selected5, setSelected1, setSelected2, setSelected3, setSelected4, setSelected5 } = useStore();

  const selectedSlots = [selected1, selected2, selected3, selected4, selected5];
  const setSelectedSlots = [setSelected1, setSelected2, setSelected3, setSelected4, setSelected5];

  const itemsPerPage = 6; // 페이지당 보여줄 아이템 수(임의)
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = products.slice(indexOfFirstItem, indexOfLastItem);
  
  useEffect(() => {
    setCurrentPage(1); // 페이지가 변경될 때마다 첫 페이지로 초기화
  }, [products, setCurrentPage]);

  const showDescription = (description) => {
    setSelectedProduct(description);
  };

  const setting = (item) => {
    // 선택된 슬롯을 찾아 상태를 업데이트
    for (let i = 0; i < userSlotNum; i++) {
      if (!selectedSlots[i].name) {
        setSelectedSlots[i](item);
        savedSlot.push(item);
        break;
      }
    }
  };
  
  const clear = (selected, clearFunction) => {
    // 선택된 슬롯의 인덱스를 찾기
    const selectedIndex = selectedSlots.findIndex(slot => slot === selected);
    // 현재 클릭한 슬롯을 비움
    clearFunction({ name: null, description: null });
    savedSlot.splice(selectedIndex, 1)[0];
    // 앞쪽 슬롯들을 앞으로 당겨오기
    for (let i = selectedIndex; i < userSlotNum - 1; i++) {
      setSelectedSlots[i](selectedSlots[i + 1]);
    }
    // 마지막 슬롯을 비움
    setSelectedSlots[userSlotNum - 1]({ name: null, description: null });
  };

  const reset = () => {
    // 화면에서 지우기
    setSelected1({ name: null, description: null });
    setSelected2({ name: null, description: null });
    setSelected3({ name: null, description: null });
    setSelected4({ name: null, description: null });
    setSelected5({ name: null, description: null });
    // 배열 비우기
    setSavedSlot([]);
  }

  console.log(savedSlot); // 왜 콘솔에 두번씩 나오는지 모르겠음 -> useEffect쓰면되는데 어차피 나중에 지울거라 그대로 두기

  return (
    <div>
      {/* 페이지네이션 */}
      <div className="flex justify-center my-2 font-cusFont1">
        <button 
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
          onClick={() => setCurrentPage(currentPage - 1)} 
          disabled={currentPage === 1} 
        >
          이전 페이지
        </button>
        <button 
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
          onClick={() => setCurrentPage(currentPage + 1)} 
          disabled={currentItems.length < itemsPerPage} 
        >
          다음 페이지
        </button>
      </div>
      {/* 사용자가 가지고 있는 아이템(products) */}
      <div className="flex justify-evenly">
        {/* {products.map((item, index) => ( */}
        {currentItems.map((item, index) => (
          <div 
          key={index} 
          className="font-cusFont1 flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white mb-6 text-center"
          // onClick={() => setting({ name: item.name, description: item.description })}
          >
            <p className="text-2xl py-4 my-2">{item.name}</p>
            <div className="h-[80px] text-xl py-3 my-2">
              <button 
                className="mx-2 px-4 py-2 bg-amber-300 hover:bg-amber-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
                onClick={() => {
                  showDescription(item.description);
                  openItemModal();
                }}
              >
                상세보기
              </button>
              <button 
                className="mx-2 px-4 py-2 bg-blue-300 hover:bg-blue-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
                onClick={() => setting({ name: item.name })}
              >
                선택하기
              </button>
            </div>
          </div>
        ))}
      </div>
      {/* 소지 슬롯개수(userSlot)만큼 */}
      <div className="flex justify-evenly mb-24">
      {[...Array(userSlotNum)].map((_, index) => {
          const selectedSlot = selectedSlots[index];
          const setSelectedSlot = setSelectedSlots[index];
          return (
            <div
              key={index}
              className="flex-grow-1 w-[200px] h-[100px]  border-2 px-6 py-4 rounded-lg border-black bg-white hover:cursor-pointer text-center"
              onClick={() => clear(selectedSlot, setSelectedSlot)}
            >
              {/* 여기에 클릭해서 넣은거 표시되어야함 */}
              <p className="font-cusFont1 text-2xl py-2 my-2">{selectedSlot.name}</p>
              {/* <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
                <p className="">{selectedSlot.description}</p>
              </div> */}
            </div>
          );
        })}
      </div>
      <div className="flex justify-center my-2">
        <button onClick={reset} className="font-cusFont1 mx-2 px-4 py-2 bg-red-300 hover:bg-red-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform">초기화</button>
      </div>
    </div>
  );
}

SlotSetting.propTypes = {
  openItemModal: PropTypes.func.isRequired,
}

export default SlotSetting;
