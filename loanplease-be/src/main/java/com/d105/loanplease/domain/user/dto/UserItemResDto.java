package com.d105.loanplease.domain.user.dto;

import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.user.entity.UserItem;
import lombok.Builder;
import lombok.Getter;

@Getter
public class UserItemResDto {

    private Long userItemId;
    private Long itemId;
    private Integer itemCount;
    private String itemName;
    private String itemContent;

    public UserItemResDto(UserItem userItem) {
        this.userItemId = userItem.getUserItemId();
        Item item = userItem.getItem();
        this.itemCount = userItem.getCount();
        this.itemId = item.getItemId();
        this.itemName = item.getName();
        this.itemContent = item.getContent();
    }
}
