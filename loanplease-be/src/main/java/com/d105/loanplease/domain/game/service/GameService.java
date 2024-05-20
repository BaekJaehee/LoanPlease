package com.d105.loanplease.domain.game.service;

import com.d105.loanplease.domain.game.dto.GameInfo;
import com.d105.loanplease.domain.game.dto.UseItemResponse;
import com.d105.loanplease.domain.game.response.GameInfoResponse;
import com.d105.loanplease.domain.game.response.ResultResponse;
import com.d105.loanplease.domain.game.response.ScoreResponse;
import org.springframework.http.ResponseEntity;

public interface GameService {
    ResponseEntity<GameInfoResponse> getGameInfo();

    ResponseEntity<ScoreResponse> getAwayCustomer(GameInfo gameInfo);
    ResponseEntity<ScoreResponse> gainScore(int num, GameInfo gameInfo);

    ResponseEntity<ResultResponse> saveScore(int score);
    ResponseEntity<UseItemResponse> useItem(Long userItemId);
}
