package com.d105.loanplease.global.util;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class BaseResponse<T> {
    @Schema(description = "상태 코드")
    private int status;

    @Schema(description = "상태 메세지")
    private String message;

    @Schema(description = "데이터")
    public T data;

    public static <T> BaseResponse createResponse(int code, String message, T data){
        return BaseResponse.builder()
                .status(code)
                .message(message)
                .data(data)
                .build();
    }

}
