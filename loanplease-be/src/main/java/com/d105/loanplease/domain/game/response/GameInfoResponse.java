package com.d105.loanplease.domain.game.response;

import com.d105.loanplease.domain.game.dto.GameInfo;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class GameInfoResponse {
    @Schema(description = "상태 코드")
    private int status;

    @Schema(description = "상태 메세지")
    private String message;

    @Schema(description = "데이터")
    public GameInfo data;

    public static GameInfoResponse createGameInfoResponse(int code, String message, GameInfo data){
        return GameInfoResponse.builder()
                .status(code)
                .message(message)
                .data(data)
                .build();
    }

}
