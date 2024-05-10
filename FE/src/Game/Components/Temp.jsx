import axios from "axios";
import useStore from "../../Store/GameStore";
import React, { useState, useEffect } from 'react';

import { postScoreRequest } from "../../API/CustomerAPI";

function Temp() {

  const { gameInfo } = useStore()

  result1 = postScoreRequest(1, gameInfo)

  return (
    <div>
      {result1}
    </div>
  );
}

export default Temp;
