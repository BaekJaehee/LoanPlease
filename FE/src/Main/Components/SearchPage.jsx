/* eslint-disable react/jsx-key */
import Userdata from "./Userdata.jsx";

const searchpagestyle = {
  width: "100%",
  height: "220px",
  display: "flex",
  justifyContent: "space-between",
  flexDirection: "column",
  margin: "10px 0px",
};

const searchlinestyle = {
  width: "100%",
  height: "10%",
  outline: "none",
  fontSize: "20px",
  fontFamily: "Orbit",
  borderBottom: "2px solid",
  marginBottom: "10px",
};

const searchresultarea = {
  width: "100%",
  height: "200px",
  overflowX: "hidden",
  overflowY: "auto",
};

const resultdummydata = [
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
];

function SearchPage() {
  return (
    <div style={searchpagestyle}>
      <input style={searchlinestyle} />
      <div style={searchresultarea}>
        {resultdummydata.map((value) => {
          return <Userdata data={value} />;
        })}
      </div>
    </div>
  );
}

export default SearchPage;
