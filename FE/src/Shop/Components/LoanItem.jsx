import { PropTypes } from "prop-types"; 
import useStore from "../../Store/ShopStore";

function LoanItem({ openModal }) {
  const { loanItems } = useStore();

  return (
    <div className="flex justify-evenly flex-wrap gap-4 my-2 text-center w-full">
      {loanItems.map((item, index) => (
        // <div key={index} className="flex-grow-1 w-[280px] border-2 bg-white px-6 py-4 rounded-lg border-black">
        <div key={index} className={`flex-grow-1 w-[280px] border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-4 rounded-lg border-black ${item.purchased == 1 && 'cursor not allowed text-gray-600'}`}>
          <p className="font-cusFont1 text-3xl py-4 my-2">{item.name}</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">{item.description}</p>
          </div>
          <button 
            className={`font-cusFont1 my-2 ${item.purchased === 1 ? 'bg-gray-300 border-gray-500' : 'bg-orange-400 hover:bg-orange-600 border-black'} border-2 border-b-4 rounded-lg px-3 py-2  text-xl w-[130px]`}
            onClick={() => openModal()}
            disabled={item.purchased == 1}
          >
            {/* 가격 */}
            {item.price}
          </button>
        </div>
      ))}
    </div>
  )
}

LoanItem.propTypes = {
  openModal: PropTypes.func.isRequired,
};

export default LoanItem;
