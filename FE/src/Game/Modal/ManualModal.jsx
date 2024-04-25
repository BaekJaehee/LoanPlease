import { PropTypes } from "prop-types";
import gaming from "../Assets/gaming.png";
import close from "../Assets/close.png";

function ManualModal({ closeModal }) {

  return(
    // <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-50 z-[99999]" onClick={closeModal}>
    <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-50 z-[99999]">
      <div className="relative w-[500px] max-w-[90%] bg-white p-6 rounded-lg shadow-lg border-2 border-black">
        <button
          className="absolute top-0 right-0 m-4 p-2 bg-white rounded-full text-gray-800"
          onClick={closeModal}
        >
          <img src={close} alt="닫기" className="w-7 h-7" />
        </button>
        <div className="text-center">
          <h2 className="text-2xl mb-4">게임 설명</h2>
          <div className="font-cusFont2 text-left m-2">
            <img src={gaming} alt="" className="mb-2" />
            <p>1. 고객의 요구사항을 확인합니다.</p>
            <p>2. 남은 시간과 점수를 볼수 있습니다.</p>
            <p>3. 고객의 정보를 참고하여 대출 상품을 추천합니다.</p>
            <p>4. 선택하면 상품의 상세정보를 확인할 수 있습니다.</p>
            <p>5. 요구사항에 알맞은 상품이라면 추천합니다. 고객이 무리한 요구를 할 경우 돌려보내고 새 고객을 받을 수 있습니다.</p>
            <p>이 설명문은 언제든지 볼 수 있습니다.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

ManualModal.propTypes = {
  closeModal: PropTypes.func.isRequired,
};

export default ManualModal;