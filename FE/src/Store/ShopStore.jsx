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
    {name: '슬롯 추가1', icon: addSlot, description: '대출 상품 슬롯을 영구적으로 1칸 추가합니다.', price: '2000', purchased: 1},
    {name: '슬롯 추가2', icon: addSlot, description: '대출 상품 슬롯을 영구적으로 1칸 추가합니다.', price: '5000', purchased: 0},
    {name: '시간 추가', icon: timeExtension, description: '게임 시간을 1분 추가합니다(한 게임 당 1회 한정).', price: '1000', purchased: 0},
    {name: 'VIP실 보내기', icon: vip, description: '상품 선택과 상관 없이 최고 점수를 받습니다(1회 한정).', price: '500', purchased: 0},
    {name: '보호막', icon: shield, description: '미리 사용 시, 감점 당할 일이 있으면 1회에 한해 방어합니다.', price: '500', purchased: 0},
  ],

  // isPurchasableSlot2: () => set({}),

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
    {name:'A대출', description: '금리가 어쩌고 설명', price: '123', purchased: 0},
    {name:'B대출', description: 'Lorem ipsum dolor sit amet,', price: '000', purchased: 0},
    {name:'C대출', description: 'consectetur adipiscing elit,', price: '456', purchased: 0},
    {name:'D대출', description: 'sed do eiusmod tempor incididunt ut', price: '789', purchased: 1},
    {name:'E대출', description: 'labore et dolore magna aliqua', price: '555', purchased: 1},
    {name:'F대출', description: '금리가 어쩌고 설명', price: '777', purchased: 0},
    {name:'G대출', description: '금리가 어쩌고 설명', price: '666', purchased: 0},
    {name:'H대출', description: '금리가 어쩌고 설명', price: '666', purchased: 0},
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
}));

export default useStore;