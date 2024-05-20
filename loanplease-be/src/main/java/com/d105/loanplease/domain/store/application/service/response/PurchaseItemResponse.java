package com.d105.loanplease.domain.store.application.service.response;

import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.user.dto.UserItemResDto;
import com.d105.loanplease.domain.user.entity.UserItem;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class PurchaseItemResponse {

    private Integer remainPoint;
    private List<UserItemResDto> userItemResDtoList = new ArrayList<>();

    public PurchaseItemResponse(Integer remainPoint, List<UserItem> userItemList) {
        this.remainPoint = remainPoint;

        for(UserItem userItem: userItemList) {
            userItemResDtoList.add(new UserItemResDto(userItem));
        }
    }
}
