import { BrowserRouter, Routes, Route } from "react-router-dom";

import Home from "./Main/index.jsx";
import Login from "./Main/Components/Login.jsx";
import Game from "./Game/Components/Game.jsx";
import Shop from "./Shop/Components/Shop.jsx";
import Temp from "./Game/Components/Temp.jsx"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/game" element={<Game />} />
        <Route path="/shop" element={<Shop />} />
        <Route path="/temp" element={<Temp />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
