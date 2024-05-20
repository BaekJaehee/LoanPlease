package com.d105.loanplease.domain.friend.dto.request;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Builder
public class FriendReq {
    private Long senderId;
    private Long receiverId;

}
