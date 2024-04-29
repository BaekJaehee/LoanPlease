import { useInfiniteQuery } from "@tanstack/react-query";
import Userdata from "./Userdata";
import { Fragment, useEffect } from "react";

export const resultdummydata = [
  {
    image: "",
    nickname: "23456",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
  {
    image: "",
    nickname: "dsafsgh",
    isFollow: true,
  },
  {
    image: "",
    nickname: "5467ujydsghfdsfgdsggfsdbfhrte",
    isFollow: false,
  },
];

function getData(page) {
  const result = [];
  for (var i = 6 * page; i < 6 * page + 6; i++) {
    result.push(resultdummydata[i]);
  }
  return result;
}

export const PageQuery = () => {
  const {
    isLoading,
    data,
    isFetching,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ["friends"],
    queryFn: ({ pageParam = 0 }) => getData(pageParam),
    getNextPageParam: (_lastPage, pages) => {
      if (pages.length < 6) {
        return pages.length + 1;
      } else return undefined;
    },
  });

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

  if (isLoading) return <div>loading...</div>;

  //버튼 눌러서 결과 확장 조회하기
  return (
    <div>
      {data &&
        data.pages?.map((group, i) => (
          <Fragment key={i}>
            {group && group?.map((unit) => <Userdata key={i} data={unit} />)}
          </Fragment>
        ))}
      <div>
        <button onClick={fetchNextPage} disabled={!hasNextPage}>
          More Results
        </button>
      </div>
      <div>{isFetching && !isFetchingNextPage ? "Fetching data..." : null}</div>
    </div>
  );

  //   //스크롤 이벤트 시 결과 확장 조회하기
  //   return (
  //     <div>
  //       {data &&
  //         data.pages?.map((group, i) => (
  //           <Fragment key={i}>
  //             {group && group?.map((unit) => <Userdata key={i} data={unit} />)}
  //           </Fragment>
  //         ))}
  //     </div>
  //   );
};
