/* eslint-disable react/prop-types */
import loading from "./assets/loading_dots.gif";
import Userdata from "./Userdata";
import { useEffect, useState } from "react";
import { friendsearch, friendsearchByname } from "../../API/API";
import { RankingStore } from "../Store";

export const PageQuery = () => {
  const [resultdata, setResultData] = useState(null);
  const inputdata = RankingStore((state) => state.inputdata);

  const getallfriends = async () => {
    friendsearch().then((result) => {
      result.dataBody && setResultData(result.dataBody.friends);
    });
  };

  const getInputResult = async (input) => {
    friendsearchByname(input).then((result) => {
      console.log(input);
      console.log(result);
      result.dataBody && setResultData(result.dataBody.friends);
    });
  };

  useEffect(() => {
    if (inputdata == "") {
      // 검색 화면 진입 시 현재 친구 목록 소환하기
      getallfriends();
    } else {
      // 입력값 존재할 시 해당 값을 포함하는 유저 목록 검색하기
      getInputResult(inputdata);
    }
  }, [inputdata]);

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
