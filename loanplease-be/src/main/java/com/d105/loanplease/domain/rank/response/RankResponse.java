package com.d105.loanplease.domain.rank.response;

import com.d105.loanplease.domain.game.dto.GameInfo;
import com.d105.loanplease.domain.rank.dto.Rank;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class RankResponse {
    @Schema(description = "상태 코드")
    private int status;

    @Schema(description = "상태 메세지")
    private String message;

    @Schema(description = "데이터")
    public List<Rank> data;

    public static RankResponse createRankResponse(int code, String message, List<Rank> data){
        return RankResponse.builder()
                .status(code)
                .message(message)
                .data(data)
                .build();
    }

}
