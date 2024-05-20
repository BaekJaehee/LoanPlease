package com.d105.loanplease.domain.friend.dto.request;

import com.d105.loanplease.domain.user.entity.Friendship;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public class FriendListReq {
    @JsonProperty("friends")
    private List<Friendship> friends;

    // Getter와 Setter
    public List<Friendship> getFriends() {
        return friends;
    }

    public void setFriends(List<Friendship> friends) {
        this.friends = friends;
    }
}
