package com.d105.loanplease.domain.game.controller;

import com.d105.loanplease.domain.game.dto.Score;
import com.d105.loanplease.domain.game.response.GameInfoResponse;
import com.d105.loanplease.domain.game.response.ResultResponse;
import com.d105.loanplease.domain.game.response.ScoreResponse;
import com.d105.loanplease.domain.game.service.GameService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@CrossOrigin("*")
@RequestMapping("/game")
public class GameController {

    private final GameService gameService;
    @Operation(summary = "게임 정보 불러오기", description = "게임에 필요한 정보(고객 정보, 금융/비금융 정보, 원하는 대출 상품)를 불러옵니다.")
    @ApiResponses({
            @ApiResponse(responseCode = "200", description = "(message : \"Success\", code : 200)",
                    content = @Content(schema = @Schema(implementation = GameInfoResponse.class)))
    })
    @GetMapping("/loanrequest")
    public ResponseEntity<GameInfoResponse> getGameInfo(){
        return gameService.getGameInfo();
    };

    @Operation(summary = "점수 획득하기", description = "적절한 점수를 리턴합니다.")
    @ApiResponses({
            @ApiResponse(responseCode = "200", description = "(message : \"Success\", code : 200)",
                    content = @Content(schema = @Schema(implementation = Score.class)))
    })
    @GetMapping("/score/{mode}")
    public ResponseEntity<ScoreResponse> gainScore(@PathVariable int mode){
        return gameService.gainScore(mode);
    };

    @Operation(summary = "점수 저장하기", description = "유저의 최고 기록보다 점수가 높으면 점수가 갱신됩니다.")
    @ApiResponses({
            @ApiResponse(responseCode = "200", description = "(message : \"Success\", code : 200)",
                    content = @Content(schema = @Schema(implementation = ResultResponse.class)))
    })
    @PatchMapping("/score")
    public ResponseEntity<ResultResponse> saveScore(int score){
        return gameService.saveScore(score);
    };
}