import { useRef } from "react";
import { PropTypes } from "prop-types"; 

function ItemModal({ closeItemModal, selectedProduct }) {
  const modalRef = useRef();

  // 모달 바깥을 클릭하면 모달이 닫히도록
  const handleOutsideClick = (e) => {
    if (modalRef.current && !modalRef.current.contains(e.target)) {
      closeItemModal();
    }
  };

  return(
    <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-50 z-20" onClick={handleOutsideClick}>
      <div ref={modalRef} className="w-[360px] h-[250px] bg-yellow-200 p-4 rounded-md text-center border-2 z-50 border-black flex flex-col justify-center">
        <div className="font-cusFont2 h-[130px] text-left text-lg">
          {/* product.description */}
          <p>상품 이름: {selectedProduct.loanName}</p>
          <p>설명: {selectedProduct.content}</p>
          <p>이자: {selectedProduct.interest}</p>
          <p>기한: {selectedProduct.period}</p>
        </div>
        <div className="flex justify-center mt-4">
          <button 
            className="mx-2 px-4 py-2 bg-red-300 hover:bg-red-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform font-cusFont1" 
            onClick={closeItemModal}
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  )
}

ItemModal.propTypes = {
  closeItemModal: PropTypes.func.isRequired,
  // selectedProduct: PropTypes.string.isRequired,
  selectedProduct: PropTypes.object.isRequired,
};

export default ItemModal;