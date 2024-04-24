import { PropTypes } from "prop-types"; 
import useStore from "../../Store/ShopStore";

function GameItem({ openModal }) {
  const { gameItems } = useStore();

  return (
    <div className="flex justify-evenly my-4 text-center">
      {gameItems.map((item, index) => (
        <div key={index} className="border-2 bg-white px-6 py-6 rounded-lg w-[280px] h-[500px] border-black">
          <img src={item.icon} alt="슬롯추가" className="w-32 h-32 mx-auto my-5" />
          <p className="font-cusFont1 py-5 my-5 text-3xl">{item.name}</p>
          <div className="font-cusFont2 py-2 my-2 mx-3 text-xl h-[90px]">
            <p>{item.description}</p>
          </div>
          <button 
            className="font-cusFont1 my-7 bg-orange-400 hover:bg-orange-600 border-2 border-b-4 rounded-lg px-3 py-2 text-xl w-[130px] border-black"
            onClick={() => openModal()}
          >
            {item.price}
          </button>
        </div>
      ))}
      
    </div>
  )
}

GameItem.propTypes = {
  openModal: PropTypes.func.isRequired,
};

export default GameItem;