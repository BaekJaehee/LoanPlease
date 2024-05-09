import { PropTypes } from "prop-types"; 
import { useEffect } from "react";
import useStore from "../../Store/ShopStore";

function LoanItem({ openShopModal, loanItems }) {
  // const { loanItems, currentPage2, setCurrentPage2 } = useStore();
  const { currentPage2, setCurrentPage2 } = useStore();

  // const itemsPerPage = 8; // 페이지당 보여줄 아이템 수(임의)
  const itemsPerPage = 4; // 페이지당 보여줄 아이템 수(임의)
  const indexOfLastItem = currentPage2 * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = loanItems.slice(indexOfFirstItem, indexOfLastItem);
  
  useEffect(() => {
    setCurrentPage2(1); // 페이지가 변경될 때마다 첫 페이지로 초기화
  // }, [loanItems, setCurrentPage2]);
  }, [setCurrentPage2]);

  console.log(loanItems)


  return (
    <div>
      {/* 페이지네이션 */}
      <div className="flex justify-center my-2 font-cusFont1">
        <button 
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
          onClick={() => setCurrentPage2(currentPage2 - 1)} 
          disabled={currentPage2 === 1} 
        >
          이전 페이지
        </button>
        <button 
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform"
          onClick={() => setCurrentPage2(currentPage2 + 1)} 
          // 다 떴는데 비어있는 다음 페이지로 넘어가지 않도록
          disabled={currentItems.length !== itemsPerPage || currentPage2 === Math.ceil(loanItems.length / itemsPerPage)} 
        >
          다음 페이지
        </button>
      </div>
      {/* <div className="flex justify-evenly flex-wrap gap-4 my-2 text-center w-full"> */}
      {/* <div className="flex justify-center gap-4 my-2 text-center w-full"> */}
      <div className="flex justify-stretch gap-4 my-2 text-center w-full">
        {/* {loanItems.map((item, index) => ( */}
        {currentItems.map((item, index) => (
          // <div key={index} className={`flex-grow-1 w-[280px] border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-4 rounded-lg border-black ${item.purchased == 1 && 'cursor not allowed text-gray-600'}`}>
          <div key={index} className={`relative flex-grow-1 w-[303px] h-[500px] border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-4 rounded-lg border-black ${item.purchased == 1 && 'cursor not allowed text-gray-600'}`}>
            <p className="font-cusFont1 text-3xl py-4 my-2">{item.name}</p>
            <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
              <p className="">{item.content}</p>
              <p className="">{item.interest}</p>
              <p className="">{item.period}</p>
            </div>
            <button 
              className={`absolute bottom-3 left-1/2 transform -translate-x-1/2 font-cusFont1 my-2 ${item.purchased === 1 ? 'bg-gray-300 border-gray-500' : 'bg-orange-400 hover:bg-orange-600 border-black'} border-2 border-b-4 rounded-lg px-3 py-2  text-xl w-[130px]`}
              onClick={() => openShopModal()}
              disabled={item.purchased == 1}
            >
              {/* 가격 */}
              {item.price}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

LoanItem.propTypes = {
  openShopModal: PropTypes.func.isRequired,
  loanItems: PropTypes.array.isRequired,
};

export default LoanItem;
