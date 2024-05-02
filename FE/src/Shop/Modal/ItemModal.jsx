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
      <div ref={modalRef} className="w-[270px] h-[180px] bg-yellow-200 p-4 rounded-md font-cusFont2 text-center border-2 z-50 border-black flex flex-col justify-center">
        <div>
          {/* product.description */}
          {selectedProduct}
        </div>
      </div>
    </div>
  )
}

ItemModal.propTypes = {
  closeItemModal: PropTypes.func.isRequired,
  selectedProduct: PropTypes.string.isRequired,
};

export default ItemModal;