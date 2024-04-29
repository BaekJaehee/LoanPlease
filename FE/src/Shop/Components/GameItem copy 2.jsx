import { PropTypes } from "prop-types"; 
import useStore from "../../Store/ShopStore";

function GameItem({ openModal }) {
  const { gameItems } = useStore();

  return (
    <div className="flex justify-evenly my-4 text-center">
      {gameItems.map((item, index) => (
        // item.name !== '슬롯 추가1' || (item.name === '슬롯 추가1' && item.purchased === 0) ? (
        (item.name !== '슬롯 추가1' && item.name !== '슬롯 추가2') || (item.name === '슬롯 추가1' && item.purchased === 0) ? (
          <div key={index} className={`border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-6 rounded-lg w-[280px] h-[500px] border-black cursor-pointer`}>
            <img src={item.icon} alt="" className={`w-32 h-32 mx-auto my-5 ${item.purchased === 1 && 'opacity-50'}`} />
            <p className="font-cusFont1 py-2 my-5 text-2xl">{item.name}</p>
            <div className="font-cusFont2 py-2 my-5 mx-3 text-lg h-[70px]">
              <p>{item.description}</p>
            </div>
            <button 
              className={`font-cusFont1 my-7 ${item.purchased === 1 ? 'bg-gray-300 border-gray-500' : 'bg-orange-400 hover:bg-orange-600 border-black'} border-2 border-b-4 rounded-lg px-3 py-2 text-xl w-[130px]`}
              onClick={() => openModal()}
              disabled={item.purchased === 1}
            >
              {item.price}
            </button>
          </div>
        ) : null
      ))}
    </div>
  );
}

GameItem.propTypes = {
  openModal: PropTypes.func.isRequired,
};

export default GameItem;
