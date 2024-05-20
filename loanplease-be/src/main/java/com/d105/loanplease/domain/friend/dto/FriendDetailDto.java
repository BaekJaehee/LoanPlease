package com.d105.loanplease.domain.friend.dto;


import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;


@Setter
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class FriendDetailDto {


    private String nickname;
    private String email;
    private String profileImg;
    private Boolean isFriend;


}
