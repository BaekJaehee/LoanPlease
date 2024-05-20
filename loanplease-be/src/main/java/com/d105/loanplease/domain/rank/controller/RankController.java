package com.d105.loanplease.domain.rank.controller;

import com.d105.loanplease.domain.rank.response.RankResponse;
import com.d105.loanplease.domain.rank.service.RankService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequiredArgsConstructor
@RequestMapping("api/rank")
public class RankController {

    private final RankService rankService;
    @Operation(summary = "전체 랭킹 불러오기", description = "전체 랭킹을 10개까지 불러옵니다.")
    @ApiResponses({
            @ApiResponse(responseCode = "200", description = "(message : \"Success\", code : 200)",
                    content = @Content(schema = @Schema(implementation = RankResponse.class)))
    })
    @GetMapping
    public ResponseEntity<RankResponse> getRanks(){
        return rankService.getAllUserRanks();
    };

    @Operation(summary = "친구 랭킹 불러오기", description = "친구 랭킹을 10개까지 불러옵니다.")
    @ApiResponses({
            @ApiResponse(responseCode = "200", description = "(message : \"Success\", code : 200)",
                    content = @Content(schema = @Schema(implementation = RankResponse.class)))
    })
    @GetMapping("/friends")
    public ResponseEntity<RankResponse> getFriendsRanks(){
        return rankService.getFriendsRanks();
    };


}
