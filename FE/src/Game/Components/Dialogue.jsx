import useStore from "../../Store/GameStore.jsx"

function Dialogue () {

  const { dialogueNum, gameInfo } = useStore()

  const { customerInfo } = gameInfo;

  let message = ""

  if (dialogueNum == 0) {
    message = customerInfo && customerInfo.customerMessage
  } else if (dialogueNum == 1) {
    message = "돌려보내기"
  } else if (dialogueNum == 2) {
    message = "추천하기"
  } else {
    message = "VIP 아이템 사용"
  }

  return (
    <p>{message}</p>
  )
}

export default Dialogue