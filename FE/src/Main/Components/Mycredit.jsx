const mycreditstyle = {
  width: "100%",
  height: "35%",
  display: "flex",
  justifyContent: "space-around",
  alignItems: "center",
};

const creditarea = {
  height: "100%",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-around",
  alignItems: "center",
};

const resultarea = {
  width: "50%",
  height: "100%",
  backgroundColor: "#FFD28F",
  display: "flex",
  justifyContent: "space-around",
  alignItems: "center",
};

const resultscorestyle = {
  zIndex: "5",
  width: "80px",
  height: "80px",
  borderRadius: "160px",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
};

const buttonarea = {
  height: "100%",
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
};

function Mycredit() {
  return (
    <div style={mycreditstyle}>
      <div style={creditarea}>
        <div style={{ fontFamily: "비트비트체v2", fontSize: "14px" }}>
          나의 신용점수는?
        </div>
        <div
          onClick={() => {}}
          style={{ fontFamily: "Orbit", fontSize: "12px", cursor: "pointer" }}
        >
          서류 제출하고 점수 확인하기
        </div>
      </div>
      <div style={resultarea}>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <div style={{ fontFamily: "비트비트체v2", fontSize: "12px" }}>
            NICE
          </div>
          <div style={{ ...resultscorestyle, backgroundColor: "#5DEBD7" }}>
            815점
          </div>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <div style={{ fontFamily: "비트비트체v2", fontSize: "12px" }}>
            KCB
          </div>
          <div style={{ ...resultscorestyle, backgroundColor: "#C5FF95" }}>
            760점
          </div>
        </div>
      </div>
      <div style={buttonarea}>
        <div
          onClick={() => {}}
          style={{
            fontFamily: "비트비트체v2",
            fontSize: "16px",
            cursor: "pointer",
            marginBottom: "10px",
          }}
        >
          Logout
        </div>
        <div
          onClick={() => {}}
          style={{
            fontFamily: "Orbit",
            fontSize: "12px",
            color: "#A9A9A9",
            cursor: "pointer",
            marginTop: "10px",
          }}
        >
          탈퇴하기
        </div>
      </div>
    </div>
  );
}

export default Mycredit;
