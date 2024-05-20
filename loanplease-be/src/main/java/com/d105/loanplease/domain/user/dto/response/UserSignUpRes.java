package com.d105.loanplease.domain.user.dto.response;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class UserSignUpRes {

    private String nickname;
    private String email;
    private String profileImg;
    private Integer score;
    private Integer slotNum;

}
