import useStore from "../../Store/ShopStore";

// 슬롯에 중복으로 못넣게 하기
// clear 왜간헐적으로 앞으로 안당겨지는지

function SlotSetting() {
  const { selected1, selected2, selected3, selected4, selected5, setSelected1, setSelected2, setSelected3, setSelected4, setSelected5 } = useStore();

  const setting = (item) => {
    if (!selected1.name) {
      setSelected1(item);
    } else if (!selected2.name) {
      setSelected2(item);
    } else if (!selected3.name) {
      setSelected3(item);
    } else if (!selected4.name) {
      setSelected4(item);
    } else if (!selected5.name) {
      setSelected5(item)
    }
  };

  const clear = (selected, clearFunction) => {
    clearFunction({ name: null, description: null });
    // 클릭한 상태의 위치를 확인하여 해당 상태를 업데이트
    if (selected === selected1) {
      setSelected1(selected2);
      setSelected2(selected3);
      setSelected3({ name: null, description: null });
    } else if (selected === selected2) {
      setSelected2(selected3);
      setSelected3({ name: null, description: null });
    } else if (selected === selected3) {
      setSelected3({ name: null, description: null });
    } else if (selected === selected4) {
      setSelected4({ name: null, description: null});
    } else if (selected === selected5) {
      setSelected5({ name: null, description: null });
    }
  };

  return (
    <div>
      {/* map써서 표시하기 */}
      {/* 1-1 */}
      <div className="flex justify-evenly">
        <div
          className="flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white mb-6 hover:cursor-pointer"
          onClick={() => setting({ name: '사용자가 가지고있는 아이템 이름', description: '설명1' })}
        >
          <p className="font-cusFont1 text-2xl py-4 my-2">사용자가 가지고있는 아이템 이름</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">설명1</p>
          </div>
        </div>
        {/* 1-2 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white mb-6"
          onClick={() => setting({ name: '아이템2', description: '설명2' })}
        >
          <p className="font-cusFont1 text-2xl py-4 my-2">아이템2</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">설명2</p>
          </div>
        </div>
        {/* 1-3 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px]  border-2 px-6 py-4 rounded-lg border-black bg-white mb-6"
          onClick={() => setting({ name: '아이템3', description: '설명3' })}
        >
          <p className="font-cusFont1 text-2xl py-4 my-2">아이템3</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">설명3</p>
          </div>
        </div>
        {/* 1-4 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px]  border-2 px-6 py-4 rounded-lg border-black bg-white mb-6"
          onClick={() => setting({ name: '아이템4', description: '설명4' })}
        >
          <p className="font-cusFont1 text-2xl py-4 my-2">아이템4</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">설명4</p>
          </div>
        </div>
        {/* 1-5 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px]  border-2 px-6 py-4 rounded-lg border-black bg-white mb-6"
          onClick={() => setting({ name: '아이템5', description: '설명5' })}
        >
          <p className="font-cusFont1 text-2xl py-4 my-2">아이템5</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">설명5</p>
          </div>
        </div>
      </div>
      <hr />
      {/* 소지 슬롯개수만큼 */}
      {/* 2-1 */}
      <div className="flex justify-evenly">
        <div
          className="flex-grow-1 w-[200px] h-[250px]  border-2 px-6 py-4 rounded-lg border-black bg-white"
          onClick={() => clear(selected1, setSelected1)}
        >
          {/* 여기에 클릭해서 넣은거 표시되어야함 */}
          <p className="font-cusFont1 text-2xl py-4 my-2">{selected1.name}</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">{selected1.description}</p>
          </div>
        </div>
        {/* 2-2 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white"
          onClick={() => clear(selected2, setSelected2)}
        >
          {/* 여기에 클릭해서 넣은거 표시되어야함 */}
          <p className="font-cusFont1 text-2xl py-4 my-2">{selected2.name}</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">{selected2.description}</p>
          </div>
        </div>
        {/* 2-3 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white"
          onClick={() => clear(selected3, setSelected3)}
        >
          {/* 여기에 클릭해서 넣은거 표시되어야함 */}
          <p className="font-cusFont1 text-2xl py-4 my-2">{selected3.name}</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">{selected3.description}</p>
          </div>
        </div>
        {/* 2-4 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white"
          onClick={() => clear(selected4, setSelected4)}
        >
          {/* 여기에 클릭해서 넣은거 표시되어야함 */}
          <p className="font-cusFont1 text-2xl py-4 my-2">{selected4.name}</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">{selected4.description}</p>
          </div>
        </div>
        {/* 2-5 */}
        <div
          className="flex-grow-1 w-[200px] h-[250px] border-2 px-6 py-4 rounded-lg border-black bg-white"
          onClick={() => clear(selected5, setSelected5)}
        >
          {/* 여기에 클릭해서 넣은거 표시되어야함 */}
          <p className="font-cusFont1 text-2xl py-4 my-2">{selected5.name}</p>
          <div className="h-[80px] font-cusFont2 text-xl py-3 my-2">
            <p className="">{selected5.description}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SlotSetting;
