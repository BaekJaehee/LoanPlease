/* eslint-disable react/prop-types */
import loading from "./assets/loading_dots.gif";
import Userdata from "./Userdata";
import { useEffect, useState } from "react";
import { friendsearch, friendsearchByname } from "../../API/API";

// export const resultdummydata = [
//   {
//     image: "",
//     nickname: "23456",
//     isFollow: false,
//   },
//   {
//     image: "",
//     nickname: "dsafsgh",
//     isFollow: true,
//   },
//   {
//     image: "",
//     nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
//     isFollow: false,
//   },
//   {
//     image: "",
//     nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
//     isFollow: false,
//   },
// ];

// 더미 데이터에서 6개씩 조회하는 함수
// function getData(page) {
//   const result = [];
//   for (var i = 6 * page; i < 6 * page + 6; i++) {
//     if (i < resultdummydata.length) result.push(resultdummydata[i]);
//   }
//   return result;
// }

export const PageQuery = (props) => {
  const [resultdata, setResultData] = useState(null);

  const getallfriends = async () => {
    friendsearch().then((result) => {
      console.log(result);
      result.dataBody && setResultData(result.dataBody);
    });
  };

  const getInputResult = async (input) => {
    friendsearchByname(input).then((result) => {
      console.log(result);
      result.dataBody && setResultData(result.dataBody);
    });
  };

  useEffect(() => {
    if (props.inputdata == "") {
      // 검색 화면 진입 시 현재 친구 목록 소환하기
      getallfriends();
    } else {
      // 입력값 존재할 시 해당 값을 포함하는 유저 목록 검색하기
      getInputResult();
    }
  }, []);

  return (
    <div>
      {resultdata ? (
        <div>
          {resultdata &&
            resultdata.map((data, i) => {
              return <Userdata key={i} data={data} />;
            })}
        </div>
      ) : (
        <div className="flex h-full w-full items-center justify-center">
          <img width={60} height={60} src={loading} />
        </div>
      )}
    </div>
  );
};
