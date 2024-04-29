/* eslint-disable react/prop-types */
// import follow from "./assets/follow.png";
// import unfollow from "./assets/unfollow.png";

const userdatastyleClass = `
flex justify-between items-center 
w-[262px] h-[55px] 
font-cusFont2 text-base 
`;

const userimagestyleClass = `
w-[50px] h-[50px] 
bg-[#D9D9D9] rounded-[100px] object-cover 
`;

const followbuttonstyleClass = `
flex justify-center items-center 
w-20 h-10 cursor-pointer font-extrabold 
bg-follow bg-cover cursor-pointer 
`;

const unfollowbuttonstyleClass = `
flex justify-center items-center 
w-20 h-10 font-semibold 
bg-unfollow bg-cover cursor-pointer 
`;

const nameClass = `
flex items-center 
w-[100px] h-full 
overflow-auto text-start 
`;

// const userdatastyle = {
//   width: "262px",
//   height: "55px",
//   fontFamily: "Orbit",
//   fontSize: "16px",
//   display: "flex",
//   justifyContent: "space-between",
//   alignItems: "center",
// };

// const userimagestyle = {
//   width: "50px",
//   height: "50px",
//   backgroundColor: "#D9D9D9",
//   borderRadius: "100px",
//   objectFit: "cover",
// };

// const followbuttonstyle = {
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
//   width: "80px",
//   height: "40px",
//   fontWeight: "800",
//   backgroundImage: `url(${follow})`,
//   cursor: "pointer",
// };

// const unfollowbuttonstyle = {
//   display: "flex",
//   justifyContent: "center",
//   alignItems: "center",
//   width: "80px",
//   height: "40px",
//   fontWeight: "600",
//   backgroundImage: `url(${unfollow})`,
//   cursor: "pointer",
// };

const isFriend = (value) => {
  return value ? (
    // <div onClick={() => {}} style={followbuttonstyle}>
    <div onClick={() => {}} className={followbuttonstyleClass}>
      #친구
    </div>
  ) : (
    // <div onClick={() => {}} style={unfollowbuttonstyle}>
    <div onClick={() => {}} className={unfollowbuttonstyleClass}>
      +친구
    </div>
  );
};

function Userdata(props) {
  return (
    // <div style={userdatastyle}>
    <div className={userdatastyleClass}>
      {/* <div style={userimagestyle}> */}
      <div className={userimagestyleClass}>
        <img src={props.data.image} />
      </div>
      <div
        // style={{
        //   display: "flex",
        //   overflow: "auto",
        //   alignItems: "center",
        //   width: "100px",
        //   height: "100%",
        //   textAlign: "start",
        // }}
        className={nameClass}
      >
        {props.data.nickname}
      </div>
      {isFriend(props.data.isFollow)}
    </div>
  );
}

export default Userdata;
