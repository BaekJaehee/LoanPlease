import { useState } from "react";

const noticebarstyleClass = `
flex flex-col justify-around items-start 
w-[300px] h-[60px] rounded-[30px] bg-white 
font-cusFont1 text-xs mb-5 px-10
`;

const buttonstyleClass = `
flex justify-center items-center 
w-[40px] h-[20px] rounded-[20px] 
font-cusFont2 text-[10px] cursor-pointer 
`;

function Noticebar() {
  const [display, setDisplay] = useState(true);
  return display ? (
    <div className={noticebarstyleClass}>
      <div>000이 친구 요청을 하였습니다</div>
      <div className="flex w-[50%] justify-around">
        <div
          onClick={() => {
            setDisplay(false);
          }}
          className={buttonstyleClass + " bg-[#fc433d]"}
        >
          수락
        </div>
        <div
          onClick={() => {
            setDisplay(false);
          }}
          className={buttonstyleClass + " bg-[#ffffff]"}
        >
          거절
        </div>
      </div>
    </div>
  ) : null;
}

export default Noticebar;
