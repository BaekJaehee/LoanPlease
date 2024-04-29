import { create } from 'zustand';

const useStore = create(set => ({
  // 시작 시 모달
  showModal: false,
  setShowModal: () => set(state => ({ showModal: !state.showModal })),

  // 시간 및 스코어
  time: 180,
  timerActive: false,

  // 타이머 시작 함수
  startTimer: () => {
    set(state => {
      if (state.timerActive) {
        clearInterval(state.timerId);
      }
      const timerId = setInterval(() => {
        set(state => {
          if (state.time > 0) {
            return { time: state.time - 1 };
          } else {
            clearInterval(state.timerId); // 타이머 중지
            return { isGameEnd: true, timerActive: false }; // 게임 종료 상태 업데이트
          }
        });
      }, 1000);
      return { timerId: timerId, timerActive: true, isGameEnd: false };
    });
  },

  score: 0,
  isGameEnd: false,

  // 금융, 비금융 구분
  isFinance: true,
  activateFinance: () => set(state => ({ isFinance: true })),
  deactivateFinance: () => set(state => ({ isFinance: false })),

  // 손님 존재 여부
  isCustomer: false,
  callCustomer: () => set(state => ({ isCustomer: true })),
  endCustomer: () => set(state => ({ isCustomer: false, selectedProduct: null })),

  // 대출 상품 목록 (지금은 샘플만 넣기)
  products: [
    {
      name: '샘플 상품1',
      option1 : '샘플 옵션1',
      option2 : '샘플 옵션2',
      option3 : '샘플 옵션3',
      option4 : '샘플 옵션4',
      bgColor : 'bg-blue-300'
    },
    {
      name: '샘플 상품2',
      option1 : '샘플 옵션1',
      option2 : '샘플 옵션2',
      option3 : '샘플 옵션3',
      option4 : '샘플 옵션4',
      bgColor : 'bg-blue-300'
    },
    {
      name: '샘플 상품3',
      option1 : '샘플 옵션1',
      option2 : '샘플 옵션2',
      option3 : '샘플 옵션3',
      option4 : '샘플 옵션4',
      bgColor : 'bg-blue-300'
    },
    {
      name: '샘플 상품4',
      option1 : '샘플 옵션1',
      option2 : '샘플 옵션2',
      option3 : '샘플 옵션3',
      option4 : '샘플 옵션4',
      bgColor : 'bg-red-300'
    },
  ],

  //선택된 대출 상품
  selectedProduct: null,
  selectProduct: (product) => set((state) => {
    if (state.isCustomer) {
      return { selectedProduct: product };
    }
    return {}
  }), 

  // 게임 중도 퇴장 창 온오프
  isGamePause: false,
  setGamePause: () => set(state => ({ isGamePause: !state.isGamePause })),

}));

export default useStore;