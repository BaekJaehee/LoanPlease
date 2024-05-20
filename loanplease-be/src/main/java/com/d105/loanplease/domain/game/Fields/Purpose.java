package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum Purpose {
    HOUSE("부동산 구매"),
    CAR("차 구매"),
    ETC("기타");

    private final String purposeKorean;
}
