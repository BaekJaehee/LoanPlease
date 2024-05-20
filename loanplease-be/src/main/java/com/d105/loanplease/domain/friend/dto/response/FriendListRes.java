package com.d105.loanplease.domain.friend.dto.response;

import com.d105.loanplease.domain.friend.dto.FriendDetailDto;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.util.List;


@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class FriendListRes {
    @JsonProperty("friends")
    private List<FriendDetailDto> friends;

}
