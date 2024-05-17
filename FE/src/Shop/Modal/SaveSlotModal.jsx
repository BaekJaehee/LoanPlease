import { useRef } from "react";
import { PropTypes } from "prop-types"; 

function SaveSlotModal({ closeSaveSlotModal }) {
  const modalRef = useRef();

  // 모달 바깥을 클릭하면 모달이 닫히도록
  const handleOutsideClick = (e) => {
    if (modalRef.current && !modalRef.current.contains(e.target)) {
      closeSaveSlotModal();
    }
  };

  return(
    <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-50 z-20" onClick={handleOutsideClick}>
      <div ref={modalRef} className="w-[270px] h-[180px] bg-yellow-200 p-4 rounded-md font-cusFont1 text-center border-2 z-50 border-black flex flex-col justify-center">
        <div>
          <div className="h-[45px] text-lg  place-content-center">
            <p>슬롯이 저장되었습니다.</p>
          </div>
          <div className="flex justify-center mt-4">
            <button 
              className="mx-2 px-4 py-2 bg-red-300 hover:bg-red-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform" 
              onClick={() => closeSaveSlotModal()}
            >
              닫기
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

SaveSlotModal.propTypes = {
  closeSaveSlotModal: PropTypes.func.isRequired,
};

export default SaveSlotModal;