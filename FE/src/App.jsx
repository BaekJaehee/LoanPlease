import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import Home from "./Main/index.jsx";
import Game from "./Game/Components/Game.jsx";
import Shop from "./Shop/Components/Shop.jsx";
import GoogleLogin from "./Main/Components/GoogleLogin.jsx";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<GoogleLogin />} />
          <Route path="/game" element={<Game />} />
          <Route path="/shop" element={<Shop />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
