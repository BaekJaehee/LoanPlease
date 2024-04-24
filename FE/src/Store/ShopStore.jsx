import { create } from 'zustand';
import addSlot from "../Shop/Assets/addSlot.png";
import timeExtension from "../Shop/Assets/timeExtension.png";

const useStore = create((set) => ({
  // 컴포넌트 토글
  currentComponent: 'gameItem',
  setCurrentComponent: (component) => set({ currentComponent: component }),

  // 게임 아이템(임시)
  gameItems: [
    {name: '슬롯 추가', icon: addSlot, description: '영구적으로 슬롯을 1칸 추가합니다.', price: '500'},
    {name: '시간 연장(n초)', icon: timeExtension, description: '게임 시간을 n초 증가합니다(1회 한정).', price: '400'},
    {name: '아이템', icon: addSlot, description: '아이템설명어쩌고', price: '000'},
    {name: '아이템', icon: addSlot, description: 'Lorem ipsum dolor sit amet,', price: '123123'},
  ],

  // 대출 아이템(임시)
  loanItems: [
    {name:'A대출', description: '금리가 어쩌고 설명', price: '123'},
    {name:'B대출', description: 'Lorem ipsum dolor sit amet,', price: '000'},
    {name:'C대출', description: 'consectetur adipiscing elit,', price: '456'},
    {name:'D대출', description: 'sed do eiusmod tempor incididunt ut', price: '789'},
    {name:'E대출', description: 'labore et dolore magna aliqua', price: '555'},
    {name:'F대출', description: '금리가 어쩌고 설명', price: '777'},
    {name:'G대출', description: '금리가 어쩌고 설명', price: '666'},
    {name:'H대출', description: '금리가 어쩌고 설명', price: '666'},
  ],

  // 구매 확인 모달
  isModalOpen: false,
  openModal: () => set({ isModalOpen: true }),
  closeModal: () => set({ isModalOpen: false, isPurchased: false }),

  // 구매확정 모달로 전환
  isPurchased: false,
  setIsPurchased: () => set({ isPurchased: true }),
}));


export default useStore;