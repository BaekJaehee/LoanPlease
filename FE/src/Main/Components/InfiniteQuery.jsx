/* eslint-disable react/prop-types */
import loading from "./assets/loading_dots.gif";
import { useInfiniteQuery } from "@tanstack/react-query";
import Userdata from "./Userdata";
import { Fragment, useEffect, useState } from "react";
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
  const {
    isLoading,
    data,
    isFetching,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ["friends"],
    // queryFn: ({ pageParam = 0 }) => getData(pageParam),
    queryFn: ({ pageParam = 0 }) => getListData(pageParam),
    getNextPageParam: (_lastPage, pages) => {
      if (pages.length < 6) {
        return pages.length + 1;
      } else return undefined;
    },
  });

  const [resultdata, setResultData] = useState(null);

  const getallfriends = async () => {
    friendsearch().then((result) => {
      result.dataBody && setResultData(result.dataBody);
    });
  };

  const getInputResult = async (input) => {
    friendsearchByname(input).then((result) => {
      result.dataBody && setResultData(result.dataBody);
    });
  };

  const getListData = async (page) => {
    if (props.inputdata == "") {
      // 검색 화면 진입 시 현재 친구 목록 소환하기
      await getallfriends();
    } else {
      // 입력값 존재할 시 해당 값을 포함하는 유저 목록 검색하기
      await getInputResult();
    }
    const result = [];
    for (var i = 6 * page; i < 6 * page + 6; i++) {
      if (i < resultdata.length) {
        const item = {
          image: resultdata[i].profileImg,
          nickname: resultdata[i].nickname,
          isFollow: resultdata[i].isFollow,
        };
        result.push(item);
      }
    }
    return result;
  };

  useEffect(() => {
    let searching = false;
    const scrollHandler = async (e) => {
      const { scrollHeight, scrollTop, clientHeight } =
        e.target.scrollingElement;
      if (!searching && scrollHeight - scrollTop <= clientHeight * 1.2) {
        searching = true;
        if (hasNextPage) await fetchNextPage();
        searching = false;
      }
    };
    document.addEventListener("scroll", scrollHandler);
    return () => {
      document.removeEventListener("scroll", scrollHandler);
    };
  }, [fetchNextPage, hasNextPage]);

  useEffect(() => {
    getallfriends();
  }, []);

  if (isLoading)
    return (
      <div className="flex h-full w-full items-center justify-center">
        <img width={60} height={60} src={loading} />
      </div>
    );

  //버튼 눌러서 결과 확장 조회하기
  return (
    <div>
      {data &&
        data.pages?.map((group, i) => (
          <Fragment key={i}>
            {group && group?.map((unit) => <Userdata key={i} data={unit} />)}
          </Fragment>
        ))}
      <div className="flex justify-center font-cusFont1">
        <button onClick={fetchNextPage} disabled={!hasNextPage}>
          {hasNextPage ? "친구 소환하기" : "모든 친구를 소환했어요!"}
        </button>
        {isFetching && !isFetchingNextPage ? "소환 중입니다..." : null}
      </div>
    </div>
  );
};
