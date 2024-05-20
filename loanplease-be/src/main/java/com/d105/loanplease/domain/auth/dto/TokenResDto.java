package com.d105.loanplease.domain.auth.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter @Setter
@Builder
public class TokenResDto {

    private String accessToken;
    private String refreshToken;
    private String grantType;

}
