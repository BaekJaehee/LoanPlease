package com.d105.loanplease.domain.game.response;

import com.d105.loanplease.domain.game.dto.Score;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ScoreResponse {
    @Schema(description = "상태 코드")
    private int status;

    @Schema(description = "상태 메세지")
    private String message;

    @Schema(description = "데이터")
    public Score data;

    public static ScoreResponse createScoreResponse(int code, String message, Score data){
        return ScoreResponse.builder()
                .status(code)
                .message(message)
                .data(data)
                .build();
    }

}
