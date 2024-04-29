import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import Home from "./Main/index.jsx";
import Login from "./Main/Components/Login.jsx";
import Game from "./Game/Components/Game.jsx";
import Shop from "./Shop/Components/Shop.jsx";
import Temp from "./Game/Components/Temp.jsx";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/game" element={<Game />} />
          <Route path="/shop" element={<Shop />} />
          <Route path="/temp" element={<Temp />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
