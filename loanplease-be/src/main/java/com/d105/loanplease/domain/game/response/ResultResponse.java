package com.d105.loanplease.domain.game.response;

import com.d105.loanplease.domain.game.dto.Score;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResultResponse {
    @Schema(description = "상태 코드")
    private int status;

    @Schema(description = "상태 메세지")
    private String message;

    @Schema(description = "데이터")
    public int data;

    public static ResultResponse createResultResponse(int code, String message, int data){
        return ResultResponse.builder()
                .status(code)
                .message(message)
                .data(data)
                .build();
    }

}
