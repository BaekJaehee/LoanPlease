import { PropTypes } from "prop-types";
import { useEffect, useState } from "react";
import { purchaseLoanItem } from "../API/ShopAPI";
import useStore from "../../Store/ShopStore";

function LoanItem({ openLoanItemModal, openWarningModal, loanItems, point, products }) {
  const { currentPage2, setCurrentPage2, setPoint, setProducts } = useStore();
  const [purchasedProducts, setPurchasedProducts] = useState([]);

  const slicedLoanItems = loanItems.slice(3, 10); // 기본 아이템(인덱스 0~2) 제외
  const itemsPerPage = 4; // 페이지당 보여줄 아이템 수(임의)
  const indexOfLastItem = currentPage2 * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = slicedLoanItems.slice(indexOfFirstItem, indexOfLastItem);
  
  useEffect(() => {
    setCurrentPage2(1); // 페이지가 변경될 때마다 첫 페이지로 초기화
  }, [loanItems, setCurrentPage2]);

  useEffect(() => {
    // products에서 구매한 상품들의 ID만 추출하여 설정
    const purchasedIds = products.map((product) => product.loanId);
    setPurchasedProducts(purchasedIds);
  }, [products]);

  const handlePurchaseLoanItem = async (item) => {
    if (point < item.price) {
      openWarningModal();
      return;
    }

    try {
      const data = await purchaseLoanItem(item.loanId);
      setPoint(data.remainPoint);

      // 상품 구매 후 products 상태 업데이트
      // const updatedProducts = [...products, item];
      const updatedProducts = [...products, { ...item, loanName: item.name }];
      setProducts(updatedProducts);

      // 구매한 상품 state에 추가
      setPurchasedProducts([...purchasedProducts, item.loanId]);

      openLoanItemModal();
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="h-full">
      {/* 페이지네이션 */}
      <div className="flex justify-center my-2 font-cusFont1 h-[8%]">
        <button
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform h-[45px]"
          onClick={() => setCurrentPage2(currentPage2 - 1)}
          disabled={currentPage2 === 1}
        >
          이전 페이지
        </button>
        <button
          className="mx-2 px-4 py-2 bg-emerald-300 hover:bg-emerald-500 rounded-md border-2 border-b-4 border-black focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform h-[45px]"
          onClick={() => setCurrentPage2(currentPage2 + 1)}
          // 다 떴는데 비어있는 다음 페이지로 넘어가지 않도록
          disabled={
            currentItems.length !== itemsPerPage ||
            currentPage2 === Math.ceil(loanItems.length / itemsPerPage)
          }
        >
          다음 페이지
        </button>
      </div>

      <div className="flex justify-stretch gap-4 my-2 text-center w-full h-[90%]">
        {currentItems.map((item, index) => (
          <div
            key={index}
            className={`relative flex-grow-1 w-[300px] h-[85%] border-2 ${
              purchasedProducts.includes(item.loanId)
                ? "bg-stone-300 cursor-not-allowed text-gray-600"
                : "bg-white"
            } px-3 py-4 rounded-lg border-black ${item.color}`}
          >
            <p className="font-cusFont1 text-2xl py-4 my-2 h-[20%]">{item.name}</p>
            <div className="h-[80%] font-cusFont2 py-3 my-2">
              <p className="mb-6 text-xl">{item.content}</p>
              <p className="text-lg">
                금리: {item.interest}
                <br />
                기간: {item.period}
                <br />
                한도액: {item.limitAmount}
              </p>
            </div>
            <button
              id={`purchase-button-${item.loanId}`}
              className={`absolute bottom-3 left-1/2 transform -translate-x-1/2 font-cusFont1 my-2 ${
                purchasedProducts.includes(item.loanId)
                  ? "bg-gray-300 border-gray-500 cursor-not-allowed"
                  : "bg-orange-400 hover:bg-orange-600 border-black"
              } border-2 border-b-4 rounded-lg px-3 py-2  text-xl w-[130px] focus:ring-4 shadow-lg transform active:scale-y-75 transition-transform`}
              onClick={() => handlePurchaseLoanItem(item)}
              disabled={purchasedProducts.includes(item.loanId)}
            >
              {item.price}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

LoanItem.propTypes = {
  openLoanItemModal: PropTypes.func.isRequired,
  openWarningModal: PropTypes.func.isRequired,
  loanItems: PropTypes.array.isRequired,
  point: PropTypes.number.isRequired,
  products: PropTypes.array.isRequired,
};

export default LoanItem;
