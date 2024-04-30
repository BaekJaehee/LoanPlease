import { create } from 'zustand';
import addSlot from "../Shop/Assets/addSlot.png";
import timeExtension from "../Shop/Assets/timeExtension.png";
import vip from "../Shop/Assets/vip.png";
import shield from "../Shop/Assets/shield.png";

const useStore = create((set) => ({
  // 컴포넌트 토글
  currentComponent: 'gameItem',
  setCurrentComponent: (component) => set({ currentComponent: component }),

  // 게임 아이템(임시)
  gameItems: [
    {id:1 ,name: '슬롯 추가1', icon: addSlot, description: '대출 상품 슬롯을 영구적으로 1칸 추가합니다.', price: '2000', purchased: 0},
    {id:2 ,name: '슬롯 추가2', icon: addSlot, description: '대출 상품 슬롯을 영구적으로 1칸 추가합니다.', price: '5000', purchased: 0},
    {id:3 ,name: '시간 추가', icon: timeExtension, description: '게임 시간을 1분 추가합니다(한 게임 당 1회 한정).', price: '1000', purchased: 0},
    {id:4 ,name: 'VIP실 보내기', icon: vip, description: '상품 선택과 상관 없이 최고 점수를 받습니다(1회 한정).', price: '500', purchased: 0},
    {id:5 ,name: '보호막', icon: shield, description: '미리 사용 시, 감점 당할 일이 있으면 1회에 한해 방어합니다.', price: '500', purchased: 0},
  ],

  // 구매 아이템 선택
  selectedItem: null,
  setSelectedItem: (item) => set({ selectedItem: item }),

  nextSlot: () => set((state) => {
    const nextSlotIndex = state.gameItems.findIndex(item => item.name === '슬롯 추가1' && item.purchased === 1);
    if (nextSlotIndex !== -1) {
      const newItems = [...state.gameItems];
      // 슬롯 추가 1 제거
      newItems.splice(nextSlotIndex, 1);
      // 슬롯 추가 1의 인덱스가 제거된 후에는 슬롯 추가 2를 추가할 때 인덱스를 조정해야 함
      const nextSlotItem = {name: '슬롯 추가2', icon: addSlot, description: '영구적으로 슬롯을 1칸 추가합니다.', price: '5000', purchased: 0};
      // 슬롯 추가 2를 슬롯 추가 1 다음에 추가
      newItems.splice(nextSlotIndex, 0, nextSlotItem);
      return { gameItems: newItems };
    }
    return state;
  }),

  // 대출 아이템(임시)
  loanItems: [
    {id:1 ,name:'A대출', description: '금리가 어쩌고 설명', price: '123', purchased: 0},
    {id:2 ,name:'B대출', description: 'Lorem ipsum dolor sit amet,', price: '000', purchased: 0},
    {id:3 ,name:'C대출', description: 'consectetur adipiscing elit,', price: '456', purchased: 0},
    {id:4 ,name:'D대출', description: 'sed do eiusmod tempor incididunt ut', price: '789', purchased: 1},
    {id:5 ,name:'E대출', description: 'labore et dolore magna aliqua', price: '555', purchased: 1},
    {id:6 ,name:'F대출', description: '금리가 어쩌고 설명', price: '777', purchased: 0},
    {id:7 ,name:'G대출', description: '금리가 어쩌고 설명', price: '666', purchased: 0},
    {id:8 ,name:'H대출', description: '금리가 어쩌고 설명', price: '666', purchased: 0},
  ],

  // 구매 확인 모달
  isModalOpen: false,
  openModal: () => set({ isModalOpen: true }),
  closeModal: () => set({ isModalOpen: false, isPurchased: false }),

  // 구매확정 모달로 전환
  isPurchasing: false,
  setIsPurchasing: () => set({ isPurchasing: true }),

  // 아이템 구매 상태 변경
  setIsPurchased: (itemName, value) => set((state) => {
    const updatedGameItems = state.gameItems.map(item => {
      if (item.name === itemName) {
        return { ...item, purchased: value };
      }
      return item;
    });

    const updatedLoanItems = state.loanItems.map(item => {
      if (item.name === itemName) {
        return { ...item, purchased: value };
      }
      return item;
    });

    return { gameItems: updatedGameItems, loanItems: updatedLoanItems };
  }),

  // 슬롯 채우기
  selected1: {name: null, description: null},
  selected2: {name: null, description: null},
  selected3: {name: null, description: null},
  selected4: {name: null, description: null},
  selected5: {name: null, description: null},
  setSelected1: (item) => set({ selected1: {name: item.name, description: item.description} }),
  setSelected2: (item) => set({ selected2: {name: item.name, description: item.description} }),
  setSelected3: (item) => set({ selected3: {name: item.name, description: item.description} }),
  setSelected4: (item) => set({ selected4: {name: item.name, description: item.description} }),
  setSelected5: (item) => set({ selected5: {name: item.name, description: item.description} }),
}));

export default useStore;