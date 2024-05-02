import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import Home from "./Main/index.jsx";
import Game from "./Game/Components/Game.jsx";
import Shop from "./Shop/Components/Shop.jsx";
import Temp from "./Game/Components/Temp.jsx";
import { CookiesProvider } from "react-cookie";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <CookiesProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/game" element={<Game />} />
            <Route path="/shop" element={<Shop />} />
            <Route path="/temp" element={<Temp />} />
          </Routes>
        </BrowserRouter>
      </CookiesProvider>
    </QueryClientProvider>
  );
}

export default App;
