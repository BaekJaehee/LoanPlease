import { PropTypes } from "prop-types"; 
import useStore from "../../Store/ShopStore";

// 유저의 슬롯개수에따라 슬롯추가1,2 띄우기

function GameItem({ openShopModal, gameItems }) {
  // const { gameItems, setSelectedItem, slots, userSlotNum } = useStore();
  const { setSelectedItem, slots, userSlotNum } = useStore();

  // const handleModalOpen = () => {
  //   setSelectedItem(item);
  //   openShopModal();
  // }

  console.log(userSlotNum)

  return (
    <div>
      <div className="h-[45.6px] my-2"></div>
      <div className="flex justify-evenly my-2 w-full text-center">
        {/* 슬롯은 따로 표시 */}
        {slots.map((item, index) => (
          (userSlotNum === 3 && index === 0) || (userSlotNum === 4 && index === 1) ? (
            <div key={index} className={`relative border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-6 rounded-lg w-[280px] h-[500px] border-black cursor-pointer`}>
              <img src={item.icon} alt="" className={`h-28 mx-auto my-5 ${item.purchased === 1 && 'opacity-50'}`} />
              <p className="font-cusFont1 py-2 my-5 text-3xl">{item.name}</p>
              <div className="font-cusFont2 py-2 my-5 mx-3 text-lg h-[70px]">
                <p>{item.description}</p>
              </div>
              <button 
                className={`absolute bottom-5 left-1/2 transform -translate-x-1/2 font-cusFont1 my-7 ${item.purchased === 1 ? 'bg-gray-300 border-gray-500' : 'bg-orange-400 hover:bg-orange-600 border-black'} border-2 border-b-4 rounded-lg px-3 py-2 text-xl w-[130px]`}
                onClick={() => {
                  setSelectedItem(item)
                  openShopModal()
                }}
                disabled={item.purchased === 1}
              >
                {item.price}
              </button>
            </div>
          ) : (userSlotNum === 5 && index === 1) ? (
            <div key={index} className="relative border-2 bg-stone-300 px-6 py-6 rounded-lg w-[280px] h-[500px] border-black cursor-not-allowed">
              <img src={item.icon} alt="" className={`h-28 mx-auto my-5 ${item.purchased === 1 && 'opacity-50'}`} />
              <p className="font-cusFont1 py-2 my-5 text-3xl">{item.name}</p>
              <div className="font-cusFont2 py-2 my-5 mx-3 text-lg h-[70px]">
                <p>{item.description}</p>
              </div>
              <button 
                className={`absolute bottom-5 left-1/2 transform -translate-x-1/2 font-cusFont1 my-7 bg-gray-400 border-gray-500 border-2 border-b-4 rounded-lg px-3 py-2 text-xl w-[130px] cursor-not-allowed`}
                disabled={true}
              >
                {item.price}
              </button>
            </div>
          ) : null
        ))}
        {gameItems.map((item, index) => (
          // <div key={index} className={`border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-6 rounded-lg w-[280px] h-[500px] border-black cursor-pointer`}>
          <div key={index} className={`relative border-2 ${item.purchased === 1 ? 'bg-stone-300' : 'bg-white'} px-6 py-6 rounded-lg w-[280px] h-[500px] border-black cursor-pointer`}>
            {/* <img src={item.icon} alt="" className={`w-32 h-32 mx-auto my-5 ${item.purchased === 1 && 'opacity-50'}`} /> */}
            {/* <img src={item.img} alt="" className={`h-28 mx-auto my-5 ${item.purchased === 1 && 'opacity-50'}`} /> */}
            {/* 프론트 내부의 이미지를 쓰기 위해서는 이미지 컬럼 추가 필요 */}
            <img src={`/src/Shop/Assets/${item.img}.png`} alt={item.img} className={`h-28 mx-auto my-5 ${item.purchased === 1 && 'opacity-50'}`} />
            <p className="font-cusFont1 py-2 my-5 text-3xl">{item.name}</p>
            <div className="font-cusFont2 py-2 my-5 mx-3 text-lg h-[70px]">
              <p>{item.content}</p>
            </div>
            <button 
              className={`absolute bottom-5 left-1/2 transform -translate-x-1/2 font-cusFont1 my-7 ${item.purchased === 1 ? 'bg-gray-300 border-gray-500' : 'bg-orange-400 hover:bg-orange-600 border-black'} border-2 border-b-4 rounded-lg px-3 py-2 text-xl w-[130px]`}
              onClick={() => {
                setSelectedItem(item)
                openShopModal()
              }}
              // onClick={handleModalOpen}
              disabled={item.purchased === 1}
            >
              {item.price}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

GameItem.propTypes = {
  openShopModal: PropTypes.func.isRequired,
  // gameItems: PropTypes.object.isRequired,
  gameItems: PropTypes.array.isRequired,
};

export default GameItem;
