package com.d105.loanplease.domain.user.dto.request;


import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class UserUpdateReq {
    private String email;
    private String nickname;
    private String profileImage;
}
