import { create } from 'zustand';
import { fetchCustomerRequest, fetchUserRequest, patchScoreRequest, postUseItem } from '../API/CustomerAPI.jsx';

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
            patchScoreRequest(state.score) // 스코어 업데이트 API 호출
              .then(response => set({ credit: response.data }))
              .catch(error => console.error('Failed to update score:', error));
            set({ isGameEnd: true, timerActive: false }); // 게임 종료 상태 업데이트
            return {};
          }
        });
      }, 1000);
      return { timerId: timerId, timerActive: true, isGameEnd: false };
    });
  },

  score: 0,
  credit: 0,
  isGameEnd: false,
  showScore: false,
  changeScore: 0,

  isVipActive: true,
  isShieldActive: true,
  isTimeActive: true,
  isShield: false,
  setIsShield: () => set(state => ({ isShield: false })),

  useVip: () => {
    set(state => {
      state.selectedProduct = null;
      state.isVipActive = false,
        state.dialogueNum = 3;
      // 버튼 비활성화
      state.isButtonEnabled = false;
      // 1초 후 재활성화 및 고객 상태 업데이트
      state.changeScore = +1000;
      state.showScore = true

      state.score = state.score + state.changeScore

      postUseItem(state.userItemId[0]).then(() => {
      }).catch(error => {
        console.error("Failed to use item:", error);
      });

      state.loadUserInfo()

      setTimeout(() => {
        set({ isButtonEnabled: true, isCustomer: false, showScore: false });
      }, 1500);
      return {};
    })
  },


  useShield: () => set(state => {
    // 우선 상태를 업데이트
    const newState = { isShieldActive: false, isShield: true };
    
    // postUseItem 함수를 호출하고 처리
    postUseItem(state.userItemId[1]).then(() => {
    }).catch(error => {
      console.error("Failed to use item:", error);
    });

    state.loadUserInfo()
  
    return newState;
  }),

  useTime: () => set(state => {
    // 우선 상태를 업데이트
    const newState = { isTimeActive: false, time: state.time + 60 };
    
    // postUseItem 함수를 호출하고 처리
    postUseItem(state.userItemId[2]).then(() => {
    }).catch(error => {
      console.error("Failed to use item:", error);
    });

    state.loadUserInfo()

    return newState;
  }),


  // 금융, 비금융 구분
  isFinance: true,
  activateFinance: () => set(state => ({ isFinance: true })),
  deactivateFinance: () => set(state => ({ isFinance: false })),

  // 손님 존재 여부
  isCustomer: false,
  callCustomer: async () => {
    set({ isCustomer: true, dialogueNum: 0 });
    await useStore.getState().loadGameInfo();  // 상태의 비동기 함수 호출
  },

  dialogueNum: 0,
  setDialogue: (newDialogueNum) => set({ dialogueNum: newDialogueNum }),

  isButtonEnabled: true,
  setButtonDisabled: () => {
    set({ isButtonEnabled: false });
    setTimeout(() => {
      set({ isButtonEnabled: true });
    }, 1000);
    set(state => ({ isCustomer: false }));
  },

  // 대출 상품 목록 (지금은 샘플만 넣기)
  products: [],

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

  gameInfo: [],

  loadGameInfo: async () => {
    try {
      const data = await fetchCustomerRequest();
      set({ gameInfo: data.data });
    } catch (error) {
      console.error('Failed to load game data:', error);
    }
  },

  updateCustomerState: (newState) => set(newState),

  userInfo: [],
  items: [0, 0, 0],
  userItemId: [0, 0, 0],

  loadUserInfo: async () => {
    try {
      const data = await fetchUserRequest();

      const newItems = [
        data.dataBody.userItemList[0].itemCount,
        data.dataBody.userItemList[1].itemCount,
        data.dataBody.userItemList[2].itemCount,
      ];

      const newIds = [
        data.dataBody.userItemList[0].userItemId,
        data.dataBody.userItemList[1].userItemId,
        data.dataBody.userItemList[2].userItemId,
      ];

      set({ userInfo: data.dataBody, products: data.dataBody.userLoanList, items: newItems, userItemId: newIds });

      console.log(userInfo)

    } catch (error) {
      console.error('Failed to load game data:', error);
    }
  },

  resetGame: () => set({
    showModal: false,
    time: 180,
    timerActive: false,
    score: 0,
    credit: 0,
    isGameEnd: false,
    showScore: false,
    changeScore: 0,
    isVipActive: true,
    isShieldActive: true,
    isTimeActive: true,
    isShield: false,
    isFinance: true,
    isCustomer: false,
    dialogueNum: 0,
    isButtonEnabled: true,
    products: [],
    selectedProduct: null,
    isGamePause: false,
    gameInfo: [],
    userInfo: [],
    items: [0, 0, 0],
    userItemId: [0, 0, 0],
  }),


}));

export default useStore;