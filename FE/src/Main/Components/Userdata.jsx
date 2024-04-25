/* eslint-disable react/prop-types */
import follow from "./assets/follow.png";
import unfollow from "./assets/unfollow.png";

const userdatastyle = {
  width: "262px",
  height: "55px",
  fontFamily: "Orbit",
  fontSize: "16px",
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};

const userimagestyle = {
  width: "50px",
  height: "50px",
  backgroundColor: "#D9D9D9",
  borderRadius: "100px",
  objectFit: "cover",
};

const followbuttonstyle = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  width: "80px",
  height: "40px",
  fontWeight: "800",
  backgroundImage: `url(${follow})`,
  cursor: "pointer",
};

const unfollowbuttonstyle = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  width: "80px",
  height: "40px",
  fontWeight: "600",
  backgroundImage: `url(${unfollow})`,
  cursor: "pointer",
};

const isFriend = (value) => {
  return value ? (
    <div onClick={() => {}} style={followbuttonstyle}>
      #친구
    </div>
  ) : (
    <div onClick={() => {}} style={unfollowbuttonstyle}>
      +친구
    </div>
  );
};

function Userdata(props) {
  return (
    <div style={userdatastyle}>
      <div style={userimagestyle}>
        <img src={props.data.image} />
      </div>
      <div
        style={{
          display: "flex",
          overflow: "auto",
          alignItems: "center",
          width: "100px",
          height: "100%",
          textAlign: "start",
        }}
      >
        {props.data.nickname}
      </div>
      {isFriend(props.data.isFollow)}
    </div>
  );
}

export default Userdata;
