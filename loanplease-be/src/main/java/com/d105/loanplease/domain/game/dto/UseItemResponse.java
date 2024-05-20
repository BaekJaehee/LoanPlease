package com.d105.loanplease.domain.game.dto;

import com.d105.loanplease.domain.user.dto.UserItemResDto;
import lombok.Builder;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
@Builder
public class UseItemResponse {
    private List<UserItemResDto> userItemResDtoList = new ArrayList<>();
}
