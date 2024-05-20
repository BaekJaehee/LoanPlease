package com.d105.loanplease.domain.rank.service;

import com.d105.loanplease.domain.rank.response.RankResponse;
import org.springframework.http.ResponseEntity;

public interface RankService {
    ResponseEntity<RankResponse> getAllUserRanks();

    ResponseEntity<RankResponse> getFriendsRanks();
}
