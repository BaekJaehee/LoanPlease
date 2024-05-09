import { useEffect, useState } from "react";
import myavatar from "./assets/myavatar.png";
import myicon from "./assets/myicon.png";
import { MypageStore } from "../Store";

const myprofilestyleClass = `
flex justify-around items-center 
w-full h-[40%] 
`;

const myavatarareaClass = `
flex flex-col justify-around items-center 
w-[20%] h-full ml-8
`;

const avatarbuttonstyleClass = `
flex justify-center items-center 
w-36 h-12 rounded-[20px] mx-5 my-2.5 
font-cusFont2 text-base text-white cursor-pointer 
bg-gradient-to-l from-[#67c6e3] to-[#378ce7] 
`;

const myinfoareaClass = `
flex flex-col justify-center items-center 
w-[60%] h-full 
`;

const linestyleClass = `
flex flex-row justify-between items-baseline 
w-[60%] h-[40%]
`;

const inputlinestyleClass = `
w-full max-w-[100px] 
text-xs font-cusFont2 
outline-none border-b-2 border-black mx-2.5 
`;

// const myprofilestyle = {
//   width: "100%",
//   height: "35%",
//   display: "flex",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const myavatararea = {
//   width: "20%",
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "space-around",
//   alignItems: "center",
// };

// const avatarbuttonstyle = {
//   width: "144px",
//   height: "48px",
//   borderRadius: "20px",
//   margin: "10px 20px",
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
//   fontFamily: "Orbit",
//   fontSize: "16px",
//   color: "#ffffff",
//   backgroundImage:
//     "linear-gradient(180deg, rgb(103, 198, 227) 0%, rgb(55, 140, 231) 100%)",
// };

// const myinfoarea = {
//   width: "60%",
//   height: "100%",
//   display: "flex",
//   flexDirection: "column",
//   justifyContent: "center",
//   alignItems: "center",
// };

// const linestyle = {
//   width: "60%",
//   margin: "10px 0px",
//   display: "flex",
//   flexDirection: "row",
//   justifyContent: "space-between",
//   alignItems: "baseline",
// };

function Myprofile() {
  const editmode = MypageStore((state) => state.editmode);
  const setEditmode = MypageStore((state) => state.setEditmode);
  const [img, setImg] = useState(`${myavatar}`);
  const [nickname, setNickname] = useState("loan_please");
  const [email, setEmail] = useState("ssafy@gmail.com");

  const changeImg = (e) => {
    const file = e.target.files[0];
    if (file === null) return;
    const url = URL.createObjectURL(file);
    if (url !== "") setImg(url);
  };

  useEffect(() => {
    if (localStorage.getItem("mydata")) {
      setImg(JSON.parse(localStorage.getItem("mydata")).image);
      setNickname(JSON.parse(localStorage.getItem("mydata")).nick);
      setEmail(JSON.parse(localStorage.getItem("mydata")).email);
    }
  }, []);

  return (
    // <div style={myprofilestyle}>
    <div className={myprofilestyleClass}>
      {/* <div style={myavatararea}> */}
      <div className={myavatarareaClass}>
        <div>
          <img
            className="max-h-[60px] max-w-[60px]"
            width={60}
            height={60}
            src={img}
          />
        </div>
        {/* <div style={avatarbuttonstyle}>아바타 바꾸기</div> */}
        {/* <div className={avatarbuttonstyleClass}>아바타 바꾸기</div> */}
        <label className={avatarbuttonstyleClass} htmlFor="imagefile">
          아바타 바꾸기
        </label>
        <input
          type="file"
          id="imagefile"
          accept="image/*"
          onChange={(e) => {
            changeImg(e);
            e.target.value = "";
          }}
          className="hidden"
        />
      </div>
      {/* <div style={myinfoarea}> */}
      <div className={myinfoareaClass}>
        {/* <div style={linestyle}> */}
        <div className={linestyleClass}>
          {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "16px" }}> */}
          <div className="min-w-[50px] font-cusFont1 text-base">닉네임</div>
          {/* <div style={{ fontFamily: "Orbit", fontSize: "12px" }}> */}
          {editmode ? (
            <input
              className={inputlinestyleClass}
              onChange={(e) => {
                setNickname(e.target.value);
              }}
              value={nickname}
            />
          ) : (
            <div className="max-w-[100px] truncate font-cusFont2 text-xs ">
              {nickname}
            </div>
          )}
          {/* <div
            style={{ fontFamily: "Orbit", fontSize: "8px", cursor: "pointer" }}
          > */}
          <div
            onClick={() => {
              setEditmode(!editmode);
            }}
            className="cursor-pointer font-cusFont2 text-[8px]"
          >
            변경하기
          </div>
        </div>
        {/* <div style={linestyle}> */}
        <div className={linestyleClass}>
          {/* <div style={{ fontFamily: "비트비트체v2", fontSize: "16px" }}> */}
          <div className="font-cusFont1 text-base">이메일</div>
          {/* <div style={{ width: "60%", fontFamily: "Orbit", fontSize: "12px" }}> */}
          <div className="w-[60%] font-cusFont2 text-xs">{email}</div>
        </div>
      </div>
      <img width={70} src={myicon} />
    </div>
  );
}

export default Myprofile;
